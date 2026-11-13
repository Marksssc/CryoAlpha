# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
#
# Modified by Mark Schuttelaars (2026): map-guided diffusion sampling
# (cryo-EM density guidance, pose fit). Based on
# alphafold3/model/network/diffusion_head.py of AlphaFold 3 v3.0.4.

"""Diffusion Head."""

from collections.abc import Callable

from alphafold3.common import base_config
from alphafold3.model import feat_batch
from alphafold3.model import model_config
from alphafold3.model.components import haiku_modules as hm
from alphafold3.model.components import utils
from alphafold3.model.network import atom_cross_attention
from alphafold3.model.network import diffusion_transformer
from alphafold3.model.network import featurization
from alphafold3.model.network import noise_level_embeddings
import haiku as hk
import jax
import jax.numpy as jnp


# Carefully measured by averaging multimer training set.
SIGMA_DATA = 16.0
FLIP_BLUR = 6.0              # Å; first flip check: when the map starts showing its shape
FLIP_BLUR_2 = 2.0            # Å; second check (step ~153, scored at ~3.5 Å blur):
                             # fine enough to tell the two ends apart
MAX_ROT_DEG_PER_STEP = 3.0   # largest angle the per-step line search tries
T_BLUR = 10.0                # Å; translation step only once blur <= this
T_MAX = 10.0                 # Å; cap on the total map-frame offset |t|


_GUIDANCE = {"total_loss": None, "guidance_strength": 0.0, "pose_loss": None,
             "rescale": True, "n_atoms": None, "gradient_mode": "denoiser",
             "active_steps": None, "random_augmentation": False, "strength_schedule_t": None,
             "verbose": False}
 
def set_guidance(
    total_loss, guidance_strength: float = 0.0, pose_loss=None,
    rescale: bool = True, n_atoms: int | None = None, gradient_mode: str = "denoiser",
    active_steps: tuple[int, int] | None = None,
    random_augmentation: bool = False, strength_schedule_t=None,
    verbose: bool = False,
) -> None:

  """Set the guidance loss/strength used by `sample()`, bypassing AF3 config.

  guidance_strength: per step strength in step size as fraction of the normal denoiser step (peak).
  pose_loss: the loss that moves the protein rigidly for the rigid fit.
  rescale: Whether the guidance is rescaled up to guidance strength.
  n_atoms: number of real atoms. If this is set correctly it only guides the atoms, saving compute.
  gradient_mode: "denoiser" (default) pulls the shape gradient back through the denoiser
  to the noisy input (J^T g, one extra denoiser forward + backward per guided step);
  "x0hat" applies the gradient taken on the denoised prediction directly.
  active_steps: (start, end) diffusion steps, where outside start <= step < end the step is
  plain unguided AF3. None = every step.
  random_augmentation: keep AF3's random rotation + translation at every step (off by
  default when guided). This can be used when one uses pose-free losses
  strength_schedule_t: optional callable t = step / n_steps -> multiplier on the strength
  (the guidance YAML's strength_schedule). None = off.
  verbose: print the per-step diagnostics ([JAC], step sizes, [POSE]) for every sample.
  Off: only the [NAN] line when the NaN guard fires.
  """
  if random_augmentation and pose_loss is not None:
    raise ValueError("random_augmentation needs pose_loss None (no pose fit)")
  # pose_loss None (`pose_loss: none` in the guidance YAML): no pose fit at all -- only for
  # orientation-free losses (p(r), domain_integrity); the structure stays centred on the map COM.
  _GUIDANCE["total_loss"] = total_loss
  _GUIDANCE["guidance_strength"] = guidance_strength
  _GUIDANCE["pose_loss"] = pose_loss
  _GUIDANCE["rescale"] = rescale
  if gradient_mode not in ("x0hat", "denoiser"):
    raise ValueError(f"unknown gradient_mode {gradient_mode!r}")
  _GUIDANCE["n_atoms"] = n_atoms
  _GUIDANCE["gradient_mode"] = gradient_mode
  _GUIDANCE["active_steps"] = None if active_steps is None else tuple(int(s) for s in active_steps)
  _GUIDANCE["random_augmentation"] = bool(random_augmentation)
  _GUIDANCE["strength_schedule_t"] = strength_schedule_t
  _GUIDANCE["verbose"] = bool(verbose)
 
 
def clear_guidance() -> None:
  """Explicitly disable guidance (safer than set_guidance(None, 0.0))."""
  _GUIDANCE["total_loss"] = None
  _GUIDANCE["guidance_strength"] = 0.0
  _GUIDANCE["strength_schedule_t"] = None
  _GUIDANCE["pose_loss"] = None
  _GUIDANCE["rescale"] = True
  _GUIDANCE["n_atoms"] = None
  _GUIDANCE["gradient_mode"] = "denoiser"
  _GUIDANCE["active_steps"] = None
  _GUIDANCE["random_augmentation"] = False
  _GUIDANCE["verbose"] = False


def random_rotation(key):
  # Create a random rotation (Gram-Schmidt orthogonalization of two
  # random normal vectors)
  v0, v1 = jax.random.normal(key, shape=(2, 3))
  e0 = v0 / jnp.maximum(1e-10, jnp.linalg.norm(v0))
  v1 = v1 - e0 * jnp.dot(v1, e0, precision=jax.lax.Precision.HIGHEST)
  e1 = v1 / jnp.maximum(1e-10, jnp.linalg.norm(v1))
  e2 = jnp.cross(e0, e1)
  return jnp.stack([e0, e1, e2])


def random_augmentation(
    rng_key: jnp.ndarray,
    positions: jnp.ndarray,
    mask: jnp.ndarray,
    apply: bool = True
) -> jnp.ndarray:
  """Apply random rigid augmentation.

  Args:
    rng_key: random key
    positions: atom positions of shape (<common_axes>, 3)
    mask: per-atom mask of shape (<common_axes>,)

  Returns:
    Transformed positions with the same shape as input positions.
  """

  if not apply:
    return positions
  
  rotation_key, translation_key = jax.random.split(rng_key)

  center = utils.mask_mean(
      mask[..., None], positions, axis=(-2, -3), keepdims=True, eps=1e-6
  )
  rot = random_rotation(rotation_key)
  translation = jax.random.normal(translation_key, shape=(3,))

  augmented_positions = (
      jnp.einsum(
          '...i,ij->...j',
          positions - center,
          rot,
          precision=jax.lax.Precision.HIGHEST,
      )
      + translation
  )
  return augmented_positions * mask[..., None]

def recenter(positions, mask):
  center = utils.mask_mean(mask[..., None], positions, axis=(-2, -3), keepdims=True, eps=1e-6)
  return (positions - center) * mask[..., None]

def noise_schedule(t, smin=0.0004, smax=160.0, p=7):
  return (
      SIGMA_DATA
      * (smax ** (1 / p) + t * (smin ** (1 / p) - smax ** (1 / p))) ** p
  )


class ConditioningConfig(base_config.BaseConfig):
  pair_channel: int
  seq_channel: int
  prob: float


class SampleConfig(base_config.BaseConfig):
  steps: int
  gamma_0: float = 0.8
  gamma_min: float = 1.0
  noise_scale: float = 1.003
  step_scale: float = 1.5
  num_samples: int = 1
  ### custom guidance
  total_loss: object | None = None
  guidance_strength: float = 0.0


class DiffusionHead(hk.Module):
  """Denoising Diffusion Head."""

  class Config(
      atom_cross_attention.AtomCrossAttEncoderConfig,
      atom_cross_attention.AtomCrossAttDecoderConfig,
  ):
    """Configuration for DiffusionHead."""

    eval_batch_size: int = 5
    eval_batch_dim_shard_size: int = 5
    conditioning: ConditioningConfig = base_config.autocreate(
        prob=0.8, pair_channel=128, seq_channel=384
    )
    eval: SampleConfig = base_config.autocreate(
        num_samples=5,
        steps=200,
    )
    transformer: diffusion_transformer.Transformer.Config = (
        base_config.autocreate()
    )

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      name='diffusion_head',
  ):
    self.config = config
    self.global_config = global_config
    super().__init__(name=name)

  @hk.transparent
  def _conditioning(
      self,
      batch: feat_batch.Batch,
      embeddings: dict[str, jnp.ndarray],
      noise_level: jnp.ndarray,
      use_conditioning: bool,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    single_embedding = use_conditioning * embeddings['single']
    pair_embedding = use_conditioning * embeddings['pair']

    rel_features = featurization.create_relative_encoding(
        seq_features=batch.token_features,
        max_relative_idx=32,
        max_relative_chain=2,
    ).astype(pair_embedding.dtype)
    features_2d = jnp.concatenate([pair_embedding, rel_features], axis=-1)
    pair_cond = hm.Linear(
        self.config.conditioning.pair_channel,
        precision='highest',
        name='pair_cond_initial_projection',
    )(
        hm.LayerNorm(
            use_fast_variance=False,
            create_offset=False,
            name='pair_cond_initial_norm',
        )(features_2d)
    )

    for idx in range(2):
      pair_cond += diffusion_transformer.transition_block(
          pair_cond, 2, self.global_config, name=f'pair_transition_{idx}'
      )

    target_feat = embeddings['target_feat']
    features_1d = jnp.concatenate([single_embedding, target_feat], axis=-1)
    single_cond = hm.LayerNorm(
        use_fast_variance=False,
        create_offset=False,
        name='single_cond_initial_norm',
    )(features_1d)
    single_cond = hm.Linear(
        self.config.conditioning.seq_channel,
        precision='highest',
        name='single_cond_initial_projection',
    )(single_cond)

    noise_embedding = noise_level_embeddings.noise_embeddings(
        sigma_scaled_noise_level=noise_level / SIGMA_DATA
    )
    single_cond += hm.Linear(
        self.config.conditioning.seq_channel,
        precision='highest',
        name='noise_embedding_initial_projection',
    )(
        hm.LayerNorm(
            use_fast_variance=False,
            create_offset=False,
            name='noise_embedding_initial_norm',
        )(noise_embedding)
    )

    for idx in range(2):
      single_cond += diffusion_transformer.transition_block(
          single_cond, 2, self.global_config, name=f'single_transition_{idx}'
      )

    return single_cond, pair_cond

  def __call__(
      self,
      # positions_noisy.shape: (num_token, max_atoms_per_token, 3)
      positions_noisy: jnp.ndarray,
      noise_level: jnp.ndarray,
      batch: feat_batch.Batch,
      embeddings: dict[str, jnp.ndarray],
      use_conditioning: bool,
  ) -> jnp.ndarray:

    with utils.bfloat16_context():
      # Get conditioning
      trunk_single_cond, trunk_pair_cond = self._conditioning(
          batch=batch,
          embeddings=embeddings,
          noise_level=noise_level,
          use_conditioning=use_conditioning,
      )

      # Extract features
      sequence_mask = batch.token_features.mask
      atom_mask = batch.predicted_structure_info.atom_mask

      # Position features
      act = positions_noisy * atom_mask[..., None]
      act = act / jnp.sqrt(noise_level**2 + SIGMA_DATA**2)

      enc = atom_cross_attention.atom_cross_att_encoder(
          token_atoms_act=act,
          trunk_single_cond=embeddings['single'],
          trunk_pair_cond=trunk_pair_cond,
          config=self.config,
          global_config=self.global_config,
          batch=batch,
          name='diffusion',
      )
      act = enc.token_act

      # Token-token attention
      act = jnp.asarray(act, dtype=jnp.float32)

      act += hm.Linear(
          act.shape[-1],
          precision='highest',
          initializer=self.global_config.final_init,
          name='single_cond_embedding_projection',
      )(
          hm.LayerNorm(
              use_fast_variance=False,
              create_offset=False,
              name='single_cond_embedding_norm',
          )(trunk_single_cond)
      )

      act = jnp.asarray(act, dtype=jnp.float32)
      trunk_single_cond = jnp.asarray(trunk_single_cond, dtype=jnp.float32)
      trunk_pair_cond = jnp.asarray(trunk_pair_cond, dtype=jnp.float32)
      sequence_mask = jnp.asarray(sequence_mask, dtype=jnp.float32)

      transformer = diffusion_transformer.Transformer(
          self.config.transformer, self.global_config
      )
      act = transformer(
          act=act,
          single_cond=trunk_single_cond,
          mask=sequence_mask,
          pair_cond=trunk_pair_cond,
      )
      act = hm.LayerNorm(
          use_fast_variance=False, create_offset=False, name='output_norm'
      )(act)
      # (n_tokens, per_token_channels)

      # (Possibly) atom-granularity decoder
      assert isinstance(enc, atom_cross_attention.AtomCrossAttEncoderOutput)
      position_update = atom_cross_attention.atom_cross_att_decoder(
          token_act=act,
          enc=enc,
          config=self.config,
          global_config=self.global_config,
          batch=batch,
          name='diffusion',
      )

      skip_scaling = SIGMA_DATA**2 / (noise_level**2 + SIGMA_DATA**2)
      out_scaling = (
          noise_level * SIGMA_DATA / jnp.sqrt(noise_level**2 + SIGMA_DATA**2)
      )
    # End `with utils.bfloat16_context()`.

    return (
        skip_scaling * positions_noisy + out_scaling * position_update
    ) * atom_mask[..., None]

### HELPER FUNCTIONS NEW
def _masked_centroid(x, m):
  return jnp.sum(x * m[:, None], 0) / jnp.maximum(jnp.sum(m), 1.0)


def _inertia(r, m):
  """Inertia tensor of the (centred) atoms r."""
  rr = jnp.sum(m * jnp.sum(r * r, -1))
  return rr * jnp.eye(3) - (r * m[:, None]).T @ r


def rigid_split(x, g, m):
  """Split per-atom gradient g into rigid-body and internal parts.

  Exact L2 projection onto rigid motions v + w x r. Returns the internal part
  and the angular direction w (rad per unit step) of the rigid part.
  """
  c = _masked_centroid(x, m)
  r = (x - c) * m[:, None]
  g = g * m[:, None]
  v = jnp.sum(g, 0) / jnp.maximum(jnp.sum(m), 1.0)
  tau = jnp.sum(jnp.cross(r, g), 0)
  w = jnp.linalg.solve(_inertia(r, m) + 1e-6 * jnp.eye(3), tau)
  g_rigid = v + jnp.cross(w, r)
  return (g - g_rigid) * m[:, None], w, v

def rotvec_to_matrix(omega):
  theta = jnp.sqrt(jnp.sum(omega ** 2) + 1e-12)
  k = omega / theta
  K = jnp.array([[0., -k[2], k[1]], [k[2], 0., -k[0]], [-k[1], k[0], 0.]])
  return jnp.eye(3) + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * K @ K


def rotate_about_centroid(x, R, m):
  c = _masked_centroid(x, m)
  return ((x - c) @ R.T + c) * m[:, None]


def flip_candidates(x, m):
  """Identity + 180-degree flips about the three principal axes of x."""
  c = _masked_centroid(x, m)
  r = (x - c) * m[:, None]
  _, axes = jnp.linalg.eigh((r.T @ r) / jnp.maximum(jnp.sum(m), 1.0))
  flips = jax.vmap(lambda e: 2.0 * jnp.outer(e, e) - jnp.eye(3))(axes.T)
  return jnp.concatenate([jnp.eye(3)[None], flips])       # (4, 3, 3), all det +1
###

def _check_atom_count(n_real, n_atoms):
  """Host-side check for the atom gather: more real atoms than expected would
  silently drop atoms from the losses, so stop the run instead."""
  if int(n_real) > int(n_atoms):
    raise RuntimeError(
        f"atom gather: the mask has {int(n_real)} real atoms but n_atoms={int(n_atoms)};"
        " rerun with --nogather_atoms and check the atom count")


def _report_nonfinite(finite, t_hat):
  """Host-side: print only when the NaN guard fires (a lax.cond under vmap runs both branches)."""
  if not bool(finite):
    print(f"[NAN] non-finite guided step at t_hat={float(t_hat):.4f}: unguided step used", flush=True)


def sample(
    denoising_step: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    batch: feat_batch.Batch,
    key: jnp.ndarray,
    config: SampleConfig,
) -> dict[str, jnp.ndarray]:
  """Sample using denoiser on batch.

  Args:
    denoising_step: the denoising function.
    batch: the batch
    key: random key
    config: config for the sampling process (e.g. number of denoising steps,
      etc.)

  Returns:
    a dict
      {
         'atom_positions': jnp.array(...)       # shape (<common_axes>, 3)
         'mask': jnp.array(...)                 # shape (<common_axes>,)
      }
    where the <common_axes> are
    (num_samples, num_tokens, max_atoms_per_token)
  """

  mask = batch.predicted_structure_info.atom_mask

  def apply_denoising_step(carry, xs):
    noise_level, step = xs                      # step: 0-based index of this denoising step
    key, positions, noise_level_prev, t = carry
    key, key_noise, key_aug = jax.random.split(key, 3)

    # check whether the predictions are guided and decide random augmentations
    is_guided = _GUIDANCE["total_loss"] is not None
    augment = (not is_guided) or _GUIDANCE["random_augmentation"]
    positions = random_augmentation(
        rng_key=key_aug, positions=positions, mask=mask, apply=augment
    )
    if is_guided and not augment:
        positions = recenter(positions=positions, mask=mask)

    gamma = config.gamma_0 * (noise_level > config.gamma_min)
    t_hat = noise_level_prev * (1 + gamma)

    noise_scale = config.noise_scale * jnp.sqrt(
        # Don't take sqrt of tiny negative number (happens when running on CPU).
        jnp.maximum(t_hat**2 - noise_level_prev**2, 0.0)
    )
    noise = noise_scale * jax.random.normal(key_noise, positions.shape)
    positions_noisy = positions + noise
    positions_denoised = denoising_step(positions_noisy, t_hat)

    grad = (positions_noisy - positions_denoised) / t_hat  
    d_t = noise_level - t_hat
    positions_out = positions_noisy + config.step_scale * d_t * grad

    ### Custom EM guidance
    guidance_is_set = _GUIDANCE["total_loss"] is not None
    active_loss = _GUIDANCE["total_loss"] if guidance_is_set else config.total_loss
    active_strength = _GUIDANCE["guidance_strength"] if guidance_is_set else config.guidance_strength
    if active_loss is not None:
        positions_unguided = positions_out
        flat_atoms = positions_denoised.reshape(-1, 3)
        flat_mask = mask.reshape(-1).astype(jnp.float32)

        to_blur = lambda s: s * SIGMA_DATA / jnp.sqrt(s**2 + SIGMA_DATA**2)
        blur = to_blur(t_hat)
        pose_loss = _GUIDANCE["pose_loss"]          # e.g. the OverlapLoss, unweighted
        shift = lambda x: x + t * flat_mask[:, None]   # map-frame offset; padding stays at 0

        # Atom gather: the losses are only applied to actual atoms and not all 24 tokens
        n_atoms = _GUIDANCE["n_atoms"]
        if n_atoms is None:
          take, loss_mask = (lambda x: x), flat_mask
        else:
          idx = jnp.nonzero(flat_mask, size=n_atoms, fill_value=-1)[0]
          take, loss_mask = (lambda x: x[idx]), (idx >= 0).astype(jnp.float32)
          jax.debug.callback(_check_atom_count, jnp.sum(flat_mask), n_atoms)
        pose_fn = lambda x, b: pose_loss(take(x), b, loss_mask)
        total_fn = lambda x, b: active_loss(take(x), b, loss_mask, frac=step / config.steps)

        if pose_loss is not None:
          # PART TO FIND THE 3D SPACE ORIENATION
          # (a) Flip check, twice: at the steps where the schedule crosses FLIP_BLUR
          cands = flip_candidates(flat_atoms, flat_mask)
          cand_losses = jax.vmap(lambda R: pose_fn(
              shift(rotate_about_centroid(flat_atoms, R, flat_mask)), blur))(cands)
          crosses = lambda b: (to_blur(noise_level_prev) > b) & (to_blur(noise_level) <= b)
          do_flip = crosses(FLIP_BLUR) | crosses(FLIP_BLUR_2)
          R_flip = jnp.where(do_flip, cands[jnp.argmin(cand_losses)], jnp.eye(3))

          # (b) Rotational step from the pose loss's rigid component (on x0-hat).
          x_pose = rotate_about_centroid(flat_atoms, R_flip, flat_mask)
          pose_grad = jax.grad(pose_fn)(shift(x_pose), blur)
          _, w, v = rigid_split(x_pose, pose_grad, flat_mask)
          w_dir = -w / jnp.maximum(jnp.linalg.norm(w), 1e-12)
          angles = jnp.deg2rad(jnp.array([0.0, 0.25, 0.5, 1.0, 2.0, MAX_ROT_DEG_PER_STEP]))
          trial = jax.vmap(lambda a: pose_fn(
              shift(rotate_about_centroid(x_pose, rotvec_to_matrix(a * w_dir), flat_mask)),
              blur))(angles)
          step_rad = angles[jnp.argmin(trial)]
          R_step = rotvec_to_matrix(step_rad * w_dir)
          R_total = jax.lax.stop_gradient(R_step @ R_flip)

          # (b2) Translation step: the map-frame offset t follows the pose loss's
          x_rot = rotate_about_centroid(x_pose, R_step, flat_mask)
          v_dir = -v / jnp.maximum(jnp.linalg.norm(v), 1e-12)
          t_steps = jnp.array([0.0, 0.25, 0.5, 1.0])
          t_trial = jax.vmap(lambda s: pose_fn(
              x_rot + (t + s * v_dir) * flat_mask[:, None], blur))(t_steps)
          t_step = jnp.where(blur <= T_BLUR, t_steps[jnp.argmin(t_trial)], 0.0)
          t_new = t + t_step * v_dir
          t_new = t_new * jnp.minimum(1.0, T_MAX / jnp.maximum(jnp.linalg.norm(t_new), 1e-12))
          t_new = jax.lax.stop_gradient(t_new)
          shift = lambda x: x + t_new * flat_mask[:, None]
        else:
          R_total = jnp.eye(3)
          t_new = t

        # NON-RIGID GUIDANCE
        # (c) Conformational guidance in the corrected pose, rigid part removed.
        x_conf = rotate_about_centroid(flat_atoms, R_total, flat_mask)
        g_x0 = jax.grad(total_fn)(shift(x_conf), blur)
        g_x0, _, _ = rigid_split(x_conf, g_x0, flat_mask)
        g_x0 = g_x0 @ R_total                           # get the guidance into the correct frame

        # Check whether the Jacobian is approximated or not
        mode = _GUIDANCE["gradient_mode"]
        verbose = _GUIDANCE["verbose"]
        if mode == "x0hat":
          loss_grad = g_x0
        elif mode == "denoiser":
          g = jax.lax.stop_gradient(g_x0.reshape(positions_noisy.shape))
          loss_grad = hk.grad(
              lambda x: jnp.vdot(g, denoising_step(x, t_hat)))(positions_noisy).reshape(-1, 3)
          flat_noisy = positions_noisy.reshape(-1, 3)
          total_norm = jnp.linalg.norm(loss_grad * flat_mask[:, None])
          internal, _, _ = rigid_split(flat_noisy, loss_grad, flat_mask)
          rigid_norm = jnp.linalg.norm(loss_grad * flat_mask[:, None] - internal)
          loss_grad = internal
          if verbose:
            jax.debug.print("[JAC] rigid part after pull-back {r:.3e} of total {n:.3e} ({f:.1%})",
                            r=rigid_norm, n=total_norm, f=rigid_norm / jnp.maximum(total_norm, 1e-30))
        else:
          raise ValueError(f"unknown gradient_mode {mode!r}")

        # Decide the step size of the guidance
        model_step_norm = jnp.linalg.norm(config.step_scale * d_t * grad)
        raw_guidance_grad_norm = jnp.linalg.norm(loss_grad)
        step_strength = active_strength
        if _GUIDANCE["strength_schedule_t"] is not None: 
          step_strength = step_strength * _GUIDANCE["strength_schedule_t"](step / config.steps)
        if _GUIDANCE["rescale"]:
          scale_factor = step_strength * model_step_norm / jnp.maximum(raw_guidance_grad_norm, 1e-8)
        else:
          scale_factor = 1.0

        loss_grad = loss_grad * scale_factor

        guidance_grad_norm = jnp.linalg.norm(loss_grad)

        if verbose:
          jax.debug.print(
              "t_hat={t_hat:.4f}  strength={s:.4f}  model_step_norm={m:.4f}  "
              "raw_guidance_grad_norm={rg:.6e}  guidance_grad_norm={g:.4f}  ratio={r:.4f}  "
              "max_atom_step={ma:.3f}",
              t_hat=t_hat, s=step_strength, m=model_step_norm, rg=raw_guidance_grad_norm,
              g=guidance_grad_norm, r=guidance_grad_norm / (model_step_norm + 1e-8),
              ma=jnp.max(jnp.linalg.norm(loss_grad, axis=-1)),
          )

        positions_out = positions_out - loss_grad.reshape(positions_out.shape)

        flat_out = positions_out.reshape(-1, 3)
        flat_out = rotate_about_centroid(flat_out, R_total, flat_mask)
        positions_out = flat_out.reshape(positions_out.shape)

        if verbose and pose_loss is not None:
          jax.debug.print("[POSE] blur={b:.2f} flip={f} rot_step_deg={d:.3f} "
                          "pose_loss={p0:.5f}->{p1:.5f} flip_losses={l} "
                          "t_step={ts:.2f} t_norm={tn:.2f} t={tv}",
                          b=blur, f=do_flip, d=jnp.rad2deg(step_rad),
                          p0=trial[0], p1=jnp.min(trial), l=cand_losses,
                          ts=t_step, tn=jnp.linalg.norm(t_new), tv=t_new)

        # NaN guard: necessary when guidance pushes atoms out of the box.
        finite = jnp.all(jnp.isfinite(positions_out)) & jnp.all(jnp.isfinite(t_new))
        jax.debug.callback(_report_nonfinite, finite, t_hat)
        positions_out = jnp.where(finite, positions_out, positions_unguided)
        t = jnp.where(finite, t_new, t)

        # Step window, check whether the guidance should be applied.
        if _GUIDANCE["active_steps"] is not None:
          s0, s1 = _GUIDANCE["active_steps"]
          active = (step >= s0) & (step < s1)
          positions_out = jnp.where(active, positions_out, positions_unguided)
          t = jnp.where(active, t, carry[3])

    ### end

    return (key, positions_out, noise_level, t), positions_out

  num_samples = config.num_samples

  noise_levels = noise_schedule(jnp.linspace(0, 1, config.steps + 1))

  key, noise_key = jax.random.split(key)
  positions = jax.random.normal(noise_key, (num_samples,) + mask.shape + (3,))
  positions *= noise_levels[0]

  init = (
      jax.random.split(key, num_samples),
      positions,
      jnp.tile(noise_levels[None, 0], (num_samples,)),
      jnp.zeros((num_samples, 3)),
  )

  apply_denoising_step = hk.vmap(
      apply_denoising_step, in_axes=(0, None), split_rng=(not hk.running_init())
  )
  steps = jnp.arange(config.steps, dtype=jnp.int32)
  result, _ = hk.scan(apply_denoising_step, init, (noise_levels[1:], steps), unroll=4)
  positions_out = result[1]

  final_dense_atom_mask = jnp.tile(mask[None], (num_samples, 1, 1))

  return {'atom_positions': positions_out, 'mask': final_dense_atom_mask}
