import torch
from time import time
from typing import Dict, Tuple

def torch_atleast1d(tensor):
    """
    Converts input to a 1D tensor if it is 0-dimensional (scalar).
    If the input is already 1D or higher, it returns the input unchanged.
    """
    if tensor.ndim == 0:
        return tensor.unsqueeze(0)
    return tensor

def set_diff_1d_torch(t1, t2, assume_unique=False):
    """
    Set difference of two 1D tensors.
    Returns the unique values in t1 that are not in t2.

    """
    if not assume_unique:
        t1 = torch.unique(t1)
        t2 = torch.unique(t2)

    mask = ~torch.isin(t1, t2)
    return t1[mask]

def view_as_windows_torch(image, shape):
    """View tensor as overlapping rectangular windows, with a given stride.

    Parameters
    ----------
    image : `~torch.Tensor`
        2d image tensor, with the last two dimensions
        being the image dimensions
    shape : tuple of int
        Shape of the window.
    stride : tuple of int
        Stride of the windows. By default it is half of the window size.

    Returns
    -------
    windows : `~torch.Tensor`
        Tensor of overlapping windows

    """
    for i, s in enumerate(shape):
        image = image.unfold(i, s, 1)
    return image


class Grappa:
    def __init__(self, kspace: torch.Tensor, kernel_size: Tuple[int, int] = (5, 5), coil_axis: int = -1) -> None:
        self.kspace = kspace
        self.kernel_size = kernel_size
        self.coil_axis = coil_axis
        self.lamda = 0.01

        self.kernel_var_dict = self.get_kernel_geometries()

    def get_kernel_geometries(self):
        self.kspace = torch.moveaxis(self.kspace, self.coil_axis, -1)

        if torch.sum((torch.abs(self.kspace[..., 0]) == 0).flatten()) == 0:
            return torch.moveaxis(self.kspace, -1, self.coil_axis)
        
        kx, ky = self.kernel_size[:]
        kx2, ky2 = int(kx/2), int(ky/2)
        nc = self.kspace.shape[-1]

        self.kspace = torch.nn.functional.pad(
            self.kspace, (0, 0, ky2, ky2, kx2, kx2), mode='constant'
        )

        mask = torch.abs(self.kspace[..., 0]) > 0

        # with NTF() as fP:
        #     P = np.memmap(fP, dtype=mask.cpu().numpy().dtype, mode='w+', shape=(
        #         mask.shape[0]-2*kx2, mask.shape[1]-2*ky2, 1, kx, ky))
        P = view_as_windows_torch(mask, (kx, ky))
        Psh = P.shape[:]  
        P = P.reshape((-1, kx, ky))

        P, iidx = torch.unique(P, return_inverse=True, dim=0)

        validP = torch.nonzero(~P[:, kx2, ky2]).squeeze()

        invalidP = torch.nonzero(torch.all(P == 0, dim=(1, 2)))
        validP = set_diff_1d_torch(validP, invalidP, assume_unique=True)

        validP = torch_atleast1d(validP)

        P = P.unsqueeze(-1).expand(-1, -1, -1, nc)

        holes_x = {}
        holes_y = {}
        for ii in validP:
            idx = torch.unravel_index(
                torch.nonzero(iidx == ii), Psh[:2]
            )
            x, y = idx[0]+kx2, idx[1]+ky2
            x = torch_atleast1d(x.squeeze())
            y = torch_atleast1d(y.squeeze())
            
            holes_x[ii.item()] = x
            holes_y[ii.item()] = y

        return {
            'patches': P,
            'patch_indices': validP,
            'holes_x': holes_x,
            'holes_y': holes_y
        }

    def compute_weights(self, calib: torch.Tensor) -> Dict[int, torch.Tensor]:
        calib = torch.moveaxis(calib, self.coil_axis, -1)
        kx, ky = self.kernel_size[:]
        kx2, ky2 = int(kx/2), int(ky/2)
        nc = calib.shape[-1]

        calib = torch.nn.functional.pad(
            calib, (0, 0, ky2, ky2, kx2, kx2), mode='constant'
        )

        
        A = view_as_windows_torch(calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))

        weights = {}

        for ii in self.kernel_var_dict['patch_indices']:
            S = torch.tensor(A[:, self.kernel_var_dict['patches'][ii, ...]])
            T = torch.tensor(A[:, kx2, ky2, :])
            ShS = S.conj().T @ S
            ShT = S.conj().T @ T
            lamda0 = self.lamda * torch.norm(ShS) / ShS.shape[0]
            weights[ii.item()] = torch.linalg.solve(
                ShS + lamda0 * torch.eye(ShS.shape[0]).to(S.device), ShT
            ).T

        return weights
    
    def apply_weights(self, kspace: torch.Tensor, weights: Dict[int, torch.Tensor]) -> torch.Tensor:
        fin_shape = kspace.shape[:]

        kspace = torch.moveaxis(kspace, self.coil_axis, -1)

        kx, ky = self.kernel_size[:]
        kx2, ky2 = int(kx/2), int(ky/2)

        
        adjx = kx % 2
        adjy = ky % 2

        kspace = torch.nn.functional.pad(
            kspace, (0, 0, ky2, ky2, kx2, kx2), mode='constant'
        )

        kspace = kspace.contiguous()
        recon = torch.zeros_like(kspace) 

        patch_indices = self.kernel_var_dict['patch_indices']

        for ii in patch_indices:
            idx = ii.item()
            
            # Get the holes_x and holes_y for this index
            holes_x = self.kernel_var_dict['holes_x'][idx]
            holes_y = self.kernel_var_dict['holes_y'][idx]
            
            # Convert holes_x and holes_y to torch tensors for batched indexing
            holes_x = torch.tensor(holes_x, device=kspace.device)
            holes_y = torch.tensor(holes_y, device=kspace.device)
            
            # Extract patches from kspace in a batched manner
            kx_range = torch.arange(-kx2, kx2 + adjx, device=kspace.device)
            ky_range = torch.arange(-ky2, ky2 + adjy, device=kspace.device)

            x_indices, y_indices = torch.meshgrid(kx_range, ky_range, indexing='ij')
            x_indices = x_indices[None, :, :] + holes_x[:, None, None]
            y_indices = y_indices[None, :, :] + holes_y[:, None, None]

            # Use advanced indexing to get all the patches for this index in one go
            S = kspace[x_indices, y_indices, :]

            # Apply patches mask
            S = S[:, self.kernel_var_dict['patches'][ii, ...]]
            
            # Apply the corresponding weight to each patch
            recon[holes_x, holes_y, :] = (weights[idx] @ S.T).T 
         

        return torch.moveaxis((recon + kspace)[kx2:-kx2, ky2:-ky2, :], -1, self.coil_axis)