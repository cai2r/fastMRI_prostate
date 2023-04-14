import h5py
import numpy as np
import xml.etree.ElementTree as etree
from typing import Dict, List, Optional, Sequence, Tuple


def load_file_T2(fname: str) -> Tuple:
    """
    Load T2 fastmri file.
    
    Parameters:
    -----------
    fname : str
        Path to the h5 fastmri file.
    
    Returns:
    --------
    Tuple
        A tuple containing the kspace, calibration_data, hdr, im_recon, and attributes of the file.
    """

    with h5py.File(fname, "r") as hf:
        kspace = hf["kspace"][:]       
        calibration_data = hf["calibration_data"][:] 
        hdr = hf["ismrmrd_header"][()]
        im_recon = hf["reconstruction_rss"][:]   
        atts = dict()
        atts['max'] = hf.attrs['max']
        atts['norm'] = hf.attrs['norm']
        atts['patient_id'] = hf.attrs['patient_id']
        atts['acquisition'] = hf.attrs['acquisition']

    return kspace, calibration_data, hdr, im_recon, atts


def load_file_dwi(fname: str) -> Tuple:
    """
    Load DWI fastmri file.
    
    Parameters:
    -----------
    fname : str
        Path to the h5 fastmri file.
    
    Returns:
    --------
    Tuple
        A tuple containing the kspace, calibration_data, hdr, and coil sensitivity maps.
    """

    with h5py.File(fname, 'r') as f:
        kspace = f['kspace'][:]
        calibration = f['calibration_data'][:]
        coil_sens_maps = f['coil_sens_maps'][:]
        #phase_corr = f['phase_correction'][:]
        
        ismrmrd_header = f['ismrmrd_header'][()]
        hdr = get_regridding_params(ismrmrd_header)
    
    return kspace, calibration, coil_sens_maps, hdr


def get_padding(hdr: str) -> float:
    """
    Extract the padding value from an XML header string.

    Parameters:
    -----------
    hdr : str
        The XML header string.

    Returns:
    --------
    float
        The padding value calculated as (x - max_enc)/2, where x is the readout dimension and 
        max_enc is the maximum phase-encoding dimension.
    """
    et_root = etree.fromstring(hdr)                                                              
    lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]                              
    enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1                              
    enc = ["encoding", "encodedSpace", "matrixSize"]                                              
    enc_x = int(et_query(et_root, enc + ["x"]))                                                  
    padding = (enc_x - enc_limits_max)/2                                                         

    return padding


def et_query(root: etree.Element, qlist: Sequence[str], namespace: str = "http://www.ismrm.org/ISMRMRD") -> str:
    """
    ElementTree query function.
    
    This function queries an XML document using ElementTree.
    
    Parameters:
    -----------
    root : Element
        Root of the XML document to search through.
    qlist : Sequence of str
        A sequence of strings for nested searches, e.g., ["Encoding", "matrixSize"].
    namespace : str, optional
        XML namespace to prepend query.
    
    Returns:
    --------
    str
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def zero_pad_kspace_hdr(hdr: str, unpadded_kspace: np.ndarray) -> np.ndarray:
    """
    Perform zero-padding on k-space data to have the same number of
    points in the x- and y-directions.

    Parameters
    ----------
    hdr : str
        The XML header string.
    unpadded_kspace : array-like of shape (sl, ro , coils, pe)
        The k-space data to be padded.

    Returns
    -------
    padded_kspace : ndarray of shape (sl, ro_padded, coils, pe_padded)
        The zero-padded k-space data, where ro_padded and pe_padded are
        the dimensions of the readout and phase-encoding directions after
        padding.

    Notes
    -----
    The padding value is calculated using the `get_padding` function, which
    extracts the padding value from the XML header string. If the difference
    between the readout dimension and the maximum phase-encoding dimension
    is not divisible by 2, the padding is applied asymmetrically, with one
    side having an additional zero-padding.

    """
    padding = get_padding(hdr)                                                                    
    if padding%2 != 0:
        padding_left = int(np.floor(padding))                                                    
        padding_right = int(np.ceil(padding))
    else:
        padding_left = int(padding)
        padding_right = int(padding)
    padded_kspace = np.pad(unpadded_kspace, ((0,0),(0,0),(0,0), (padding_left,padding_right)))     

    return padded_kspace


def get_regridding_params(hdr: str) -> Dict:
    """
    Extracts regridding parameters from header XML string.

    Parameters
    ----------
    hdr : str
        Header XML string.

    Returns
    -------
    dict
        A dictionary containing the extracted parameters.

    """
    res = {
        'rampUpTime': None,
        'rampDownTime': None,
        'flatTopTime': None,
        'acqDelayTime': None,
        'echoSpacing': None
    }
    
    et_root = etree.fromstring(hdr)
    namespace = {'ns': "http://www.ismrm.org/ISMRMRD"}

    for node in et_root.findall('ns:encoding/ns:trajectoryDescription/ns:userParameterLong', namespace):
        if node[0].text in res.keys():
            res[node[0].text] = float(node[1].text)
    
    return res


def save_recon(outp_dict: Dict[str, any], output_path: str) -> None:
    """
    Save reconstruction results to an HDF5 file.

    Parameters
    ----------
    outp_dict : dict
        A dictionary containing the reconstructed images, with the image names as keys.
    output_path : str
        The file path to save the reconstructed images.

    Returns
    -------
    None
    """

    hf = h5py.File(output_path, "w")
    for key, outp in outp_dict.items():
        hf.create_dataset(key, data=outp)
    hf.close()  
