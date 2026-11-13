from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List
import jax
import jax.numpy as jnp

### aggregator class
@dataclass
class TotalLoss:
    def __init__(self, losses: list, schedulers: list):
        self.losses = losses
        self.schedulers = schedulers

    def __call__(self, structure, noise_level, atom_mask=None, frac=None):
        # atom_mask (N_atoms,): real atoms vs padded slots; forwarded to every loss.
        # Schedules with input_type "t" read frac = step / n_steps (the fraction of the
        # trajectory done, 0 = start); all others read noise_level (the blur the sampler passes).
        total = 0.0
        for loss, sched in zip(self.losses, self.schedulers):
            if getattr(sched, "input_type", None) == "t":
                if frac is None:
                    raise ValueError(f"{type(sched).__name__} (input_type='t') needs the diffusion step")
                weight = sched(frac)
            else:
                weight = sched(noise_level)
            total += weight * loss(structure, noise_level, atom_mask)
            
        return total

    def grad(self, structure: jnp.ndarray, noise_level: jnp.ndarray, atom_mask=None) -> jnp.ndarray:
        return jax.grad(self.__call__)(structure, noise_level, atom_mask)

    def loss_and_grad(self, structure: jnp.ndarray, noise_level: jnp.ndarray, atom_mask=None):
        return jax.value_and_grad(self.__call__)(structure, noise_level, atom_mask)