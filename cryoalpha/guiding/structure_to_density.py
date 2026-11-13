import jax
import jax.numpy as jnp

def compute_effective_k(target_resolution: float, noise_level: float | jnp.ndarray) -> jnp.ndarray:
    """Computes Gaussian decay factor k from map resolution and noise level."""
    sigma_res = target_resolution / (jnp.pi * jnp.sqrt(2.0))
    sigma_eff = jnp.sqrt(sigma_res**2 + noise_level**2)
    return 1.0 / (2.0 * sigma_eff**2)


def simulate_density(
    structure: jnp.ndarray,
    voxel_coords: jnp.ndarray,
    target_resolution: float,
    noise_level: float | jnp.ndarray = 0.0,
    atomic_weights: jnp.ndarray | None = None,
    chunk_size: int = 16384,
) -> jnp.ndarray:
    """
    Simulates continuous Gaussian density at specific voxel coordinates from atomic coordinates.

    Args:
        structure: (N_atoms, 3) coordinates.
        voxel_coords: (M_voxels, 3) coordinates.
        target_resolution: Map resolution in Angstroms.
        noise_level: Noise variance added in quadrature to resolution sigma.
        atomic_weights: (N_atoms,) optional per-atom weights.
        chunk_size: Chunk size along voxels for jax.lax.map to control VRAM.
    """
    voxel_coords = voxel_coords.reshape(-1, 3)
    n_voxels = voxel_coords.shape[0]

    k = compute_effective_k(target_resolution, noise_level)
    w = 1.0 if atomic_weights is None else atomic_weights[None, :]

    @jax.checkpoint
    def chunk_fn(coords_chunk):
        diff = coords_chunk[:, None, :] - structure[None, :, :]
        dist_sq = jnp.sum(diff**2, axis=-1)
        return jnp.sum(w * jnp.exp(-k * dist_sq), axis=-1)

    pad_len = (-n_voxels) % chunk_size
    padded = jnp.pad(voxel_coords, ((0, pad_len), (0, 0)))
    chunks = padded.reshape(-1, chunk_size, 3)

    sim = jax.lax.map(chunk_fn, chunks).reshape(-1)[:n_voxels]
    return sim