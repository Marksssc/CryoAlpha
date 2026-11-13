from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from cryoalpha.guiding.structure_to_density import simulate_density
from cryoalpha.maps import DensityMap, pair_bin_table, pair_function_from_grid

class OverlapLoss:
    def __init__(
        self,
        density_map: "DensityMap",
        resolution: float = 3.0,
        atomic_weights: jnp.ndarray | None = None,
        chunk_size: int = 16384,
        eps: float = 1e-8,
    ):
        self.density_map = density_map                 
        self.voxel_coords = density_map.voxel_coords.reshape(-1, 3)
        self.target_resolution = resolution
        self.atomic_weights = atomic_weights
        self.chunk_size = chunk_size
        self.eps = eps

    def _normalize(self, arr: jnp.ndarray) -> jnp.ndarray:
        return (arr - jnp.mean(arr)) / (jnp.std(arr) + self.eps)

    def simulate_density(
        self,
        structure: jnp.ndarray,
        noise_level: float | jnp.ndarray = 0.0,
        atom_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        # atom_mask (N_atoms,): 1 for real atoms, 0 for padded / non-existent
        # slots, which the denoiser leaves at exactly (0, 0, 0). They must add no
        # density, or a fake peak appears at the centre of the map.
        weights = self.atomic_weights
        if atom_mask is not None:
            mask = atom_mask.astype(structure.dtype)
            weights = mask if weights is None else weights * mask
        return simulate_density(
            structure=structure,
            voxel_coords=self.voxel_coords,
            target_resolution=self.target_resolution,
            noise_level=noise_level,
            atomic_weights=weights,
            chunk_size=self.chunk_size,
        )

    def loss(
        self, structure: jnp.ndarray, noise_level: float, atom_mask: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        sim_norm = self._normalize(self.simulate_density(structure, noise_level, atom_mask))
        _, target_weights = self.density_map.blur_map(noise_level=noise_level)  
        target_norm = self._normalize(target_weights)                         
        return 1.0 - jnp.mean(sim_norm * target_norm)

    def grad(
        self, structure: jnp.ndarray, noise_level: float | jnp.ndarray, atom_mask: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        return jax.grad(self.loss)(structure, noise_level, atom_mask)

    def loss_and_grad(
        self, structure: jnp.ndarray, noise_level: float | jnp.ndarray, atom_mask: jnp.ndarray | None = None
    ):
        return jax.value_and_grad(self.loss)(structure, noise_level, atom_mask)

    def __call__(
        self, structure: jnp.ndarray, noise_level: float | jnp.ndarray, atom_mask: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        return self.loss(structure, noise_level, atom_mask)


@dataclass
class PairDistanceLoss:
    density_map: "DensityMap"
    resolution: float = 4.0
    extra_blur: float = 3.0            # Å, same on both sides
    chunk_size: int = 16384
    stride: int = 2                    # p(r) grid = every 2nd voxel (2.67 Å here); fine at a 3 Å blur

    def __post_init__(self):
        # Map side and binning table: constant, computed once.
        self.coords, self.grid_shape, voxel_zyx = self.density_map.pair_grid(self.stride)
        self.r, self.table = pair_bin_table(self.grid_shape, voxel_zyx)
        self.p_map = self.density_map.get_pair_function(self.table, self.extra_blur, stride=self.stride)

    def __call__(self, structure, noise_level, atom_mask=None):
        # Structure side: every step, on the same coarse points as the map.
        weights = None if atom_mask is None else atom_mask.astype(structure.dtype)
        sim = simulate_density(structure, self.coords, self.resolution,
                               noise_level=self.extra_blur, atomic_weights=weights,
                               chunk_size=self.chunk_size)
        g = sim.reshape(self.grid_shape)
        p_struct = pair_function_from_grid(g, self.density_map.cut_fraction, self.table)
        return jnp.sum((self.p_map - p_struct) ** 2)

@dataclass
class DomainIntegrityLoss:
    ca_gathered: jnp.ndarray     # (P, 2) atom indices of each pair, gathered layout
    ca_slots: jnp.ndarray        # (P, 2) same pairs, 24-slot layout
    n_gathered: int
    d_ref: jnp.ndarray           # (P,) reference distances (Å)
    tol: jnp.ndarray             # (P,) tolerance per pair (Å): flat, or from the PAE

    def __call__(self, structure, noise_level, atom_mask=None):
        pairs = self.ca_gathered if structure.shape[0] == self.n_gathered else self.ca_slots
        d = jnp.linalg.norm(structure[pairs[:, 0]] - structure[pairs[:, 1]], axis=-1)
        return jnp.mean(jnp.maximum(0.0, jnp.abs(d - self.d_ref) - self.tol) ** 2)
