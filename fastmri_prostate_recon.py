import argparse
from pathlib import Path

from fastmri_prostate.reconstruction.t2.prostate_t2_recon import t2_reconstruction
from fastmri_prostate.reconstruction.dwi.prostate_dwi_recon import dwi_reconstruction
from fastmri_prostate.data.mri_data import load_file_T2, load_file_dwi, save_recon

def main(args):
    Path(args['output_path']).mkdir(exist_ok=True)

    data_file_paths_dict = {}
    for split in ['test', 'training', 'validation']:
        files_by_sequence = {}
        for sequence in ['T2', 'DIFFUSION']:
            inp_path = Path(args['data_path']) / split / sequence
            outp_path = Path(args['output_path']) / split / sequence

            Path(outp_path).mkdir(parents=True, exist_ok=True)

            h5_files = []
            for fname in sorted(Path(inp_path).glob('*.h5')):
                h5_files.append({
                    'input_h5': fname,
                    'output_h5': Path(outp_path) / fname.name,
                })
            
            files_by_sequence[sequence] = h5_files
        data_file_paths_dict[split] = files_by_sequence

    # reconstruction for one case
    t2_file_path = data_file_paths_dict['training']['T2'][0]
    dwi_file_path = data_file_paths_dict['training']['DIFFUSION'][0]

    if args['sequence'] == 't2':
        kspace, calibration_data, hdr, image_recon, image_atts = load_file_T2(t2_file_path['input_h5'])
        img_dict = t2_reconstruction(kspace, calibration_data, hdr)
        save_recon(img_dict, t2_file_path['output_h5'])
    
    elif args['sequence'] == 'dwi':
        kspace, calibration, coil_sens_maps, hdr = load_file_dwi(dwi_file_path['input_h5'])
        img_dict = dwi_reconstruction(kspace, calibration, coil_sens_maps, hdr)
        save_recon(img_dict, dwi_file_path['output_h5'])
    
    elif args['sequence'] == 'both':
        kspace, calibration_data, hdr, image_recon, image_atts = load_file_T2(t2_file_path['input_h5'])
        img_dict = t2_reconstruction(kspace, calibration_data, hdr)
        save_recon(img_dict, t2_file_path['output_h5'])

        del kspace, calibration_data, hdr

        kspace, calibration_data, coil_sens_maps, hdr = load_file_dwi(dwi_file_path['input_h5'])
        img_dict = dwi_reconstruction(kspace, calibration_data, coil_sens_maps, hdr)
        save_recon(img_dict, dwi_file_path['output_h5'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prostate T2/DWI reconstruction')
    parser.add_argument(
        '--data_path', 
        type=str, 
        required=True, 
        help="Path to the dataset"
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        required=True, 
        help="Path to save the reconstructions to"
    )
    parser.add_argument(
        '--sequence', 
        default='both',
        type=str, 
        required=True,
        choices=['t2', 'dwi', 'both'],
        help="t2 or dwi or both"
    )
    args = vars(parser.parse_args())
    main(args)