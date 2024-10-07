import torch

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
        grid_mat (torch.Tensor): The gridding matrix.

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
        grid_mat = torch.eye(int(i_pts_readout), int(adc_nos))
        return grid_mat
    
    t_step = t_adcdur/(adc_nos-1)

    tt = torch.linspace(t_delay, t_delay + t_adcdur, int(adc_nos))
    kk = torch.zeros(int(adc_nos))

    for zz in range(int(adc_nos)):
        if tt[zz] < t_rampup:
            kk[zz] = (0.5/t_rampup) * torch.square(tt[zz])
        elif tt[zz] > (t_rampup + t_flattop):
            kk[zz] = (0.5/t_rampup) * torch.square(t_rampup) + (tt[zz] - t_rampup) - (0.5/t_rampdown) * (torch.square(tt[zz] - t_rampup - t_flattop))
        else:
            kk[zz] = (0.5/t_rampup) * torch.square(t_rampup) + (tt[zz] - t_rampup)

    kk = kk - kk[int(torch.floor(torch.tensor(adc_nos/2)))-1]
    need_kk = torch.linspace(kk[0], kk[-1], int(i_pts_readout))
    delta_k = need_kk[1] - need_kk[0]

    density = torch.diff(kk)
    density = torch.cat((density, density[:1]))

    grid_mat = torch.sinc(
        (torch.tile(need_kk.unsqueeze(1), (1, int(adc_nos))) - torch.tile(kk.unsqueeze(0), (int(i_pts_readout), 1)))/delta_k
    )

    grid_mat = torch.tile(density.unsqueeze(0), (int(i_pts_readout), 1)) * grid_mat
    grid_mat = grid_mat/(1e-12 + torch.tile(torch.sum(grid_mat, axis=1, keepdim=True), (1, int(adc_nos))))

    return grid_mat


def trapezoidal_regridding(img, epi_params):
    """
    Perform trapezoidal regridding on an image.

    Parameters:
    -----------
        img : (torch.Tensor)
            3D array of the input undersampled image.
        epi_params : (dict)
            A dictionary of EPI sequence parameters.
    
    Returns:
    --------        
        torch.Tensor: A 3D array representing the regridded image.

    """
    # breakpoint()
    s = img.shape
    
    os_factor = 2
    keep_oversampling = True
    
    grid_mat = get_grid_mat(epi_params, os_factor, keep_oversampling)
    grid_mat = grid_mat.float()
    
    img2 = img.permute(1, 2, 0)
    s2 = img2.shape
    img2 = img2.reshape(img2.shape[0], -1)
    
    # breakpoint()
    grid_mat = torch.tensor(grid_mat, dtype=img2.dtype).to(img2.device)
    img_out = grid_mat @ img2
    img_out = img_out.reshape(s2)
    
    img_out = img_out.permute(2, 0, 1)
    return img_out