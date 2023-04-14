import numpy as np
from tempfile import NamedTemporaryFile as NTF
from typing import Dict, Tuple
from skimage.util import view_as_windows


class Grappa:
    def __init__(self, kspace: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), coil_axis: int = -1) -> None:
        self.kspace = kspace
        self.kernel_size = kernel_size
        self.coil_axis = coil_axis
        self.lamda = 0.01

        self.kernel_var_dict = self.get_kernel_geometries()

    def get_kernel_geometries(self):
        """
        Extract unique kernel geometries based on a slice of kspace data

        Returns
        -------
        geometries : dict
            A dictionary containing the following keys:
            - 'patches': an array of overlapping patches from the k-space data.
            - 'patch_indices': an array of unique patch indices.
            - 'holes_x': a dictionary of x-coordinates for holes in each patch.
            - 'holes_y': a dictionary of y-coordinates for holes in each patch.

        Notes
        -----
        This function extracts unique kernel geometries from a slice of k-space data.
        The geometries correspond to overlapping patches that contain at least one hole.
        A hole is defined as a region of k-space data where the absolute value of the
        complex signal is equal to zero. The function returns a dictionary containing
        information about the patches and holes, which can be used to compute weights
        for each geometry using the GRAPPA algorithm.

        """
        self.kspace = np.moveaxis(self.kspace, self.coil_axis, -1)

        # Quit early if there are no holes
        if np.sum((np.abs(self.kspace[..., 0]) == 0).flatten()) == 0:
            return np.moveaxis(self.kspace, -1, self.coil_axis)
        
        kx, ky = self.kernel_size[:]
        kx2, ky2 = int(kx/2), int(ky/2)
        nc = self.kspace.shape[-1]

        self.kspace = np.pad(
            self.kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant'
        )

        mask = np.ascontiguousarray(np.abs(self.kspace[..., 0]) > 0)

        with NTF() as fP:
            # Get all overlapping patches from the mask
            P = np.memmap(fP, dtype=mask.dtype, mode='w+', shape=(
                mask.shape[0]-2*kx2, mask.shape[1]-2*ky2, 1, kx, ky))
            P = view_as_windows(mask, (kx, ky))
            Psh = P.shape[:]  # save shape for unflattening indices later
            P = P.reshape((-1, kx, ky))

            # Find the unique patches and associate them with indices
            P, iidx = np.unique(P, return_inverse=True, axis=0)

            # Filter out geometries that don't have a hole at the center.
            # These are all the kernel geometries we actually need to
            # compute weights for.
            validP = np.argwhere(~P[:, kx2, ky2]).squeeze()

            # ignore empty patches
            invalidP = np.argwhere(np.all(P == 0, axis=(1, 2)))
            validP = np.setdiff1d(validP, invalidP, assume_unique=True)

            validP = np.atleast_1d(validP)

            # Give P back its coil dimension
            P = np.tile(P[..., None], (1, 1, 1, nc))

            holes_x = {}
            holes_y = {}
            for ii in validP:
                # x, y define where top left corner is, so move to ctr,
                # also make sure they are iterable by enforcing atleast_1d
                idx = np.unravel_index(
                    np.argwhere(iidx == ii), Psh[:2]
                )
                x, y = idx[0]+kx2, idx[1]+ky2
                x = np.atleast_1d(x.squeeze())
                y = np.atleast_1d(y.squeeze())

                holes_x[ii] = x
                holes_y[ii] = y

        return {
            'patches': P,
            'patch_indices': validP,
            'holes_x': holes_x,
            'holes_y': holes_y
        }

    def compute_weights(self, calib: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute the GRAPPA weights for each slice in the input calibration data.

        Parameters:
        ----------
        calib : numpy.ndarray 
            Calibration data with shape (Nx, Nc, Ny) where Nx, Ny are the size of the image in the x and y dimensions, 
            respectively, and Nc is the number of coils.

        Returns:
        -------
        weights : dict
            A dictionary of GRAPPA weights for each patch index.

        Notes:
        -----
        The GRAPPA algorithm is used to estimate the missing k-space data in undersampled MRI acquisitions. 
        The algorithm used to compute the GRAPPA weights involves first extracting patches from the calibration data, 
        and then solving a linear system to estimate the weights. The resulting weights are stored in a dictionary 
        where the key is the patch index. The equation to solve for the weights involves taking the product of the 
        sources and the targets in the patch domain, and then regularizing the matrix using Tikhonov regularization. 
        The function uses numpy's `memmap` to store temporary files to avoid overwhelming memory usage.
        """

        calib = np.moveaxis(calib, self.coil_axis, -1)
        kx, ky = self.kernel_size[:]
        kx2, ky2 = int(kx/2), int(ky/2)
        nc = calib.shape[-1]

        calib = np.pad(
            calib, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant'
        )

        # Store windows in temporary files so we don't overwhelm memory
        with NTF() as fA:
            # Get all overlapping patches of ACS
            try:
                A = np.memmap(fA, dtype=calib.dtype, mode='w+', shape=(
                        calib.shape[0]-2*kx, calib.shape[1]-2*ky, 1, kx, ky, nc
                    ))
                A[:] = view_as_windows(calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))
            except ValueError:
                A = view_as_windows(calib, (kx, ky, nc)).reshape((-1, kx, ky, nc))

            weights = {}

            for ii in self.kernel_var_dict['patch_indices']:
                # Get the sources by masking all patches of the ACS and
                # get targets by taking the center of each patch. Source
                # and targets will have the following sizes:
                #     S : (# samples, N possible patches in ACS)
                #     T : (# coils, N possible patches in ACS)
                # Solve the equation for the weights: using numpy.linalg.solve, 
                # and Tikhonov regularization for better conditioning:
                #     SW = T
                #     S^HSW = S^HT
                #     W = (S^HS)^-1 S^HT
                #  -> W = (S^HS + lamda I)^-1 S^HT

                S = A[:, self.kernel_var_dict['patches'][ii, ...]]
                T = A[:, kx2, ky2, :]
                ShS = S.conj().T @ S
                ShT = S.conj().T @ T
                lamda0 = self.lamda*np.linalg.norm(ShS)/ShS.shape[0]
                weights[ii] = np.linalg.solve(
                    ShS + lamda0*np.eye(ShS.shape[0]), ShT
                ).T

        return weights
    
    def apply_weights(self, kspace: np.ndarray, weights: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Applies the computed GRAPPA weights to the k-space data.

        Parameters:
        ----------
            kspace : numpy.ndarray
                The k-space data to apply the weights to.

            weights : dict
                A dictionary containing the GRAPPA weights to apply.

        Returns:
        -------
            numpy.ndarray: The reconstructed data after applying the weights.
        """

        fin_shape = kspace.shape[:]

        # Put the coil dimension at the end
        kspace = np.moveaxis(kspace, self.coil_axis, -1)

        # Get shape of kernel
        kx, ky = self.kernel_size[:]
        kx2, ky2 = int(kx/2), int(ky/2)

        # adjustment factor for odd kernel size
        adjx = np.mod(kx, 2)
        adjy = np.mod(ky, 2)

        # Pad kspace data
        kspace = np.pad(  
            kspace, ((kx2, kx2), (ky2, ky2), (0, 0)), mode='constant'
        )

        with NTF() as frecon:
            # Initialize recon array
            recon = np.memmap(
                frecon, dtype=kspace.dtype, mode='w+',
                shape=kspace.shape
            )

            for ii in self.kernel_var_dict['patch_indices']:
                for xx, yy in zip(self.kernel_var_dict['holes_x'][ii], self.kernel_var_dict['holes_y'][ii]):
                    # Collect sources for this hole and apply weights
                    S = kspace[xx-kx2:xx+kx2+adjx, yy-ky2:yy+ky2+adjy, :]
                    S = S[self.kernel_var_dict['patches'][ii, ...]]
                    recon[xx, yy, :] = (weights[ii] @ S[:, None]).squeeze()

            return np.moveaxis((recon[:] + kspace)[kx2:-kx2, ky2:-ky2, :], -1, self.coil_axis)