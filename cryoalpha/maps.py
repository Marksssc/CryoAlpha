import os

import mrcfile
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple

# Voxels at or below DEFAULT_CUT_FRACTION * (max voxel value) are treated as
# noise and left out of the pair distance distribution p(r).
DEFAULT_CUT_FRACTION = 0.2


class DensityMap:
    def __init__(
        self,
        filepath: str,
        pad_voxels: int = 20,
        blur_kernel_radius: int = 64,
        cut_fraction: float = DEFAULT_CUT_FRACTION,
    ):
        self.filepath = filepath
        self.cut_fraction = cut_fraction
        self.pad_voxels = pad_voxels
        self.blur_kernel_radius = blur_kernel_radius
        
        with mrcfile.open(filepath, permissive=True) as mrc:
            raw = np.array(mrc.data, dtype=np.float32)
            # MRC data dims are (sections, rows, columns), lying along the
            # axes named by maps/mapr/mapc (1=x, 2=y, 3=z). This class assumes
            # [z][y][x]; maps written with another order would otherwise be
            # read transposed -- mirrored, if two axes are swapped.
            axis_of_dim = [int(mrc.header.maps), int(mrc.header.mapr), int(mrc.header.mapc)]
            if sorted(axis_of_dim) != [1, 2, 3]:
                raise ValueError(
                    f"{filepath}: invalid MRC axis order mapc/mapr/maps="
                    f"{axis_of_dim[::-1]}; expected a permutation of 1, 2, 3"
                )
            raw = np.transpose(raw, [axis_of_dim.index(a) for a in (3, 2, 1)])
            self.voxel_size = np.array(
                [mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], 
                dtype=np.float32
            )
            self.origin = np.array(
                [mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z], 
                dtype=np.float32
            )

        self.padded_data = np.pad(
            np.maximum(raw, 0.0), 
            pad_width=self.pad_voxels, 
            mode='constant', 
            constant_values=0.0
        )

        self._compute_locked_com()
        self._precompute_grid_coords()
        self._compute_cut()

        self.padded_data_jax = jnp.array(self.padded_data, dtype=jnp.float32)
        self.voxel_coords, self.voxel_weights = self.blur_map(noise_level=0.0)

    def _compute_locked_com(self):
        nz, ny, nx = self.padded_data.shape
        z_idx, y_idx, x_idx = np.indices((nz, ny, nx))

        x_phys = (x_idx - self.pad_voxels) * self.voxel_size[0] + self.origin[0]
        y_phys = (y_idx - self.pad_voxels) * self.voxel_size[1] + self.origin[1]
        z_phys = (z_idx - self.pad_voxels) * self.voxel_size[2] + self.origin[2]

        total_mass = np.sum(self.padded_data) + 1e-8
        com_x = np.sum(x_phys * self.padded_data) / total_mass
        com_y = np.sum(y_phys * self.padded_data) / total_mass
        com_z = np.sum(z_phys * self.padded_data) / total_mass

        self.com = np.array([com_x, com_y, com_z], dtype=np.float32)

    def _precompute_grid_coords(self):
        """Physical coords for every grid voxel — fixed, computed once. No mask:
        every step uses this same (N, 3) coordinate set; only the sampled
        values change with noise_level."""
        nz, ny, nx = self.padded_data.shape
        z_idx, y_idx, x_idx = np.indices((nz, ny, nx))

        x_phys = (x_idx - self.pad_voxels) * self.voxel_size[0] + self.origin[0]
        y_phys = (y_idx - self.pad_voxels) * self.voxel_size[1] + self.origin[1]
        z_phys = (z_idx - self.pad_voxels) * self.voxel_size[2] + self.origin[2]

        phys_coords = np.stack([x_phys, y_phys, z_phys], axis=-1).reshape(-1, 3)
        centered_coords = phys_coords - self.com
        self._grid_coords_np = centered_coords.astype(np.float32)
        self._grid_coords_jax = jnp.array(centered_coords, dtype=jnp.float32)

    def _compute_cut(self):
        """Checks the cut (voxels at or below cut_fraction * max count as noise in p(r))."""
        if not 0.0 <= self.cut_fraction < 1.0:
            raise ValueError(f"cut_fraction must be in [0, 1), got {self.cut_fraction}")
        if not (self.padded_data > self.cut_fraction * self.padded_data.max()).any():
            raise ValueError("The map has no voxels above the cut -- is it empty?")


    @staticmethod
    def _gaussian_kernel1d(sigma_voxels: jnp.ndarray, radius: int) -> jnp.ndarray:
        x = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
        k = jnp.exp(-0.5 * (x / jnp.maximum(sigma_voxels, 1e-4)) ** 2)
        return k / jnp.sum(k)

    def _blur_axis(self, grid: jnp.ndarray, kernel: jnp.ndarray, axis: int) -> jnp.ndarray:
        pad = self.blur_kernel_radius
        pad_width = [(0, 0)] * grid.ndim
        pad_width[axis] = (pad, pad)
        padded = jnp.pad(grid, pad_width, mode='constant')
        conv1d = lambda line: jnp.convolve(line, kernel, mode='valid')
        return jnp.apply_along_axis(conv1d, axis, padded)

    def blur_map(self, noise_level: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JAX-native, jittable — safe to call inside hk.scan with a traced
        noise_level. Blurs by noise_level ALONE (target-map side; resolution
        blur belongs to structure_to_density.py on the structure side).
        Fixed grid, no thresholding — shape is (nz*ny*nx,) every call."""
        avg_voxel_size = jnp.mean(jnp.asarray(self.voxel_size))
        sigma_voxels = jnp.asarray(noise_level) / avg_voxel_size

        kernel = self._gaussian_kernel1d(sigma_voxels, self.blur_kernel_radius)

        grid = self.padded_data_jax
        for axis in range(3):
            grid = self._blur_axis(grid, kernel, axis)

        weights = jnp.maximum(grid, 0.0).reshape(-1)
        return self._grid_coords_jax, weights

    def pair_grid(self, stride: int = 2):
        """Coarse grid for p(r): every `stride`-th voxel of the padded grid.
        Returns (coords (M, 3) in the same centred frame as voxel_coords,
        grid shape [z][y][x], voxel size per axis (z, y, x) in Å). The map and
        the structure are both sampled at exactly these points."""
        nz, ny, nx = self.padded_data.shape
        coords = self._grid_coords_np.reshape(nz, ny, nx, 3)[::stride, ::stride, ::stride]
        voxel_zyx = tuple(float(v) * stride for v in self.voxel_size[::-1])
        return jnp.asarray(coords.reshape(-1, 3)), coords.shape[:3], voxel_zyx

    def get_pair_function(self, table, extra_blur=3.0, cut_fraction=None, stride: int = 2):
        """Map side of the p(r) loss: blur, subsample like pair_grid, then
        pair_function_from_grid. Constant -- compute once."""
        cut = self.cut_fraction if cut_fraction is None else cut_fraction
        _, w = self.blur_map(noise_level=extra_blur)
        g = w.reshape(self.padded_data.shape)[::stride, ::stride, ::stride]
        return pair_function_from_grid(g, cut, table)


def pair_bin_table(shape, voxel_zyx, bin_width=2.0, r_max=160.0):
    """Lookup table for binning the autocorrelation by shift length. Built once
    (numpy). Returns (bin_centres, table (n_bins, max_count) int32)."""
    full = [2 * n for n in shape]
    lz, ly, lx = (np.fft.fftfreq(n) * n for n in full)
    r = np.sqrt((lz[:, None, None] * voxel_zyx[0]) ** 2 + (ly[None, :, None] * voxel_zyx[1]) ** 2
                + (lx[None, None, :] * voxel_zyx[2]) ** 2).ravel()
    edges = np.arange(0.0, r_max + bin_width, bin_width)
    n_bins = len(edges) - 1
    b = np.digitize(r, edges) - 1
    idx = np.nonzero((b >= 0) & (b < n_bins))[0]
    b = b[idx]
    order = np.argsort(b, kind="stable")
    idx, b = idx[order], b[order]
    counts = np.bincount(b, minlength=n_bins)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    table = np.full((n_bins, counts.max()), r.size, dtype=np.int32)   # dummy -> the appended 0
    table[b, np.arange(len(b)) - starts[b]] = idx
    return 0.5 * (edges[1:] + edges[:-1]), jnp.asarray(table)


def pair_function_from_grid(g, cut, table):
    """Shared by map and structure: cut -> autocorrelation (FFT) -> bin -> normalize.
    `table` must come from pair_bin_table for this grid shape."""
    g = jnp.where(g > cut * g.max(), g, 0.0)
    shape = [2 * s for s in g.shape]                                 # zero-pad: no wrap
    F = jnp.fft.rfftn(g, s=shape, axes=(0, 1, 2))
    ac = jnp.fft.irfftn(jnp.abs(F) ** 2, s=shape, axes=(0, 1, 2))
    flat = jnp.concatenate([jnp.maximum(ac, 0.0).ravel(), jnp.zeros((1,), ac.dtype)])
    p = flat[table].sum(axis=1)                                      # gather + row sum
    return p / p.sum()