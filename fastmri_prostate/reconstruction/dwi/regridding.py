import numpy as np

def get_grid_mat(epi_params, os_factor, keep_oversampling):
    """
    Generate a matrix for gridding reconstruction.

    Parameters:
    -----------
        epi_params : (dict)
            Dictionary containing EPI sequence parameters.
        os_factor : (float)
            Oversampling factor for the readout direction.
        keep_oversampling : (bool)
            Flag to keep the readout direction oversampling.
        
    Returns:
    --------
        grid_mat (numpy.ndarray): The gridding matrix.

    """
    
    t_rampup = epi_params['rampUpTime']
    t_rampdown = epi_params['rampDownTime']
    t_flattop = epi_params['flatTopTime']
    t_delay = epi_params['acqDelayTime']

    adc_nos = 200.0
    t_adcdur = 580.0

    if keep_oversampling:
        i_pts_readout = adc_nos
    else:
        i_pts_readout = adc_nos/os_factor

    if t_rampup == 0:
        grid_mat = np.eye(i_pts_readout, adc_nos)
        return
    
    t_step = t_adcdur/(adc_nos-1)

    tt = np.linspace(t_delay, t_delay + t_adcdur, int(adc_nos))
    kk = np.zeros(shape=(int(adc_nos)))

    for zz in range(int(adc_nos)):
        if tt[zz] < t_rampup:
            kk[zz] = (0.5/t_rampup) * np.square(tt[zz])
        elif tt[zz] > (t_rampup + t_flattop):
            kk[zz] = (0.5/t_rampup) * np.square(t_rampup) + (tt[zz] - t_rampup) - (0.5/t_rampdown) * (np.square(tt[zz] - t_rampup - t_flattop))
        else:
            kk[zz] = (0.5/t_rampup) * np.square(t_rampup) + (tt[zz] - t_rampup)

    kk = kk - kk[int(np.floor(adc_nos/2))-1]
    need_kk = np.linspace(kk[0], kk[len(kk)-1], int(i_pts_readout))
    delta_k = need_kk[1] - need_kk[0]

    density = np.diff(kk)
    density = np.append(density, density[0])

    grid_mat = np.sinc(
        (np.tile(need_kk, (int(adc_nos), 1)).T - np.tile(kk, (int(i_pts_readout), 1)))/delta_k
    )

    grid_mat = np.tile(density, (int(i_pts_readout), 1)) * grid_mat
    grid_mat = grid_mat/(1e-12 + np.tile(np.sum(grid_mat, axis=1), (int(adc_nos), 1)).T)

    return grid_mat


def trapezoidal_regridding(img, epi_params):
    """
    Perform trapezoidal regridding on an image.

    Parameters:
    -----------
        img : (np.ndarray)
            3D array of the input undersampled image.
        epi_params : (dict)
            A dictionary of EPI sequence parameters.
    
    Returns:
    --------        
        np.ndarray: A 3D array representing the regridded image.

    """
    s = img.shape
    
    os_factor = 2
    keep_oversampling = True
    
    grid_mat = get_grid_mat(epi_params, os_factor, keep_oversampling)
    grid_mat = grid_mat.astype('float32')
    
    img2 = np.transpose(img, (1, 2, 0))
    s2 = img2.shape
    img2 = np.reshape(img2, (img2.shape[0], np.prod(img2.shape[1:])))
    
    img_out = grid_mat @ img2
    img_out = np.reshape(img_out, s2)
    
    img_out = np.transpose(img_out, (2, 0, 1))
    return img_out

