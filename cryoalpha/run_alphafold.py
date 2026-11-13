"""CryoAlpha: cryo-EM map-guided AlphaFold 3 driver (diffusion guidance).

Monkeypatches AF3's diffusion sampler, then hands off to AF3's own run_alphafold.py
main():

  * alphafold3.model.network.diffusion_head.{DiffusionHead, sample} with the
    map-guided version from cryoalpha/af3_patches/diffusion_head.py (guidance is
    applied to the denoised coordinates at every diffusion step).
"""

import os
import sys

from absl import app
from absl import flags

# 1. Make the cryoalpha package importable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 2. Point Python at the AF3 checkout. This must point to the AlphaFold 3 root from deepmind not this version.
af3_prefix = os.environ.get("ALPHAFOLD_PREFIX")
if not af3_prefix or not os.path.isfile(os.path.join(af3_prefix, "run_alphafold.py")):
  sys.exit("Set ALPHAFOLD_PREFIX to your AlphaFold 3 (v3.0.4) checkout, the folder "
           "that contains DeepMind's run_alphafold.py.")
if af3_prefix not in sys.path:
  sys.path.insert(0, af3_prefix)

# The patch replaces AF3 3.0.4's diffusion head; other versions may differ internally.
_AF3_DIFFUSION_HEAD_MD5 = "61a9f947d7c3185badfafce5e103165c"  # AF3 v3.0.4

# 3. Patch the diffusion head before anything.
import alphafold3.model.network.diffusion_head as af3_diff_head
import hashlib
with open(af3_diff_head.__file__, "rb") as _f:
  if hashlib.md5(_f.read()).hexdigest() != _AF3_DIFFUSION_HEAD_MD5:
    print("[WARNING] AlphaFold 3's diffusion_head.py differs from v3.0.4, which this "
          "patch was written for. Results may be wrong; use AF3 v3.0.4.")
from cryoalpha.af3_patches import diffusion_head as guided_head

print(f"[PRE-PATCH] Original DiffusionHead: {getattr(af3_diff_head, 'DiffusionHead', None)}")

af3_diff_head.DiffusionHead = guided_head.DiffusionHead
af3_diff_head.sample = guided_head.sample

print("[PATCH] Successfully patched DiffusionHead and sample() at the source!")

from cryoalpha.maps import DEFAULT_CUT_FRACTION, DensityMap
from cryoalpha.guiding.setup_guidance import (
    build_strength_schedule_t, build_total_loss, gather_atom_count, load_guidance_config)
from cryoalpha import run_record

_DENSITY_MAP_PATH = flags.DEFINE_string(
    "density_map_path",
    None,
    "Path to a .mrc density map to guide the prediction towards. Leave unset"
    " to run AF3 unguided.",
)
_GUIDE_DIFFUSION = flags.DEFINE_boolean(
    "guide_diffusion",
    True,
    "Apply density-map guidance to the diffusion sampler's denoising steps"
    " (needs --density_map_path).",
)
_MAP_CUT_FRACTION = flags.DEFINE_float(
    "map_cut_fraction",
    DEFAULT_CUT_FRACTION,
    "Voxels at or below this fraction of the map's maximum are treated as noise"
    " and left out of the map's pair distance distribution p(r).",
)
_MAP_RESOLUTION_ANGSTROMS = flags.DEFINE_float(
    "map_resolution_angstroms",
    None,
    "Nominal resolution of the density map, in angstroms. Currently unused:"
    " the losses take their resolution from --guidance_config (default 4.0 A).",
)

# Diffusion guidance
_DIFFUSION_GUIDANCE_STRENGTH = flags.DEFINE_float(
    "diffusion_guidance_strength",
    0.10,
    "Per-step guidance displacement, as a fraction of the denoiser's own step norm"
    " (needs --rescale_guidance, the default). 0.10 is the tested value: below ~0.05"
    " samples stay in AF3's state, at 0.15 many blow apart.",
)
_RESCALE_GUIDANCE = flags.DEFINE_boolean(
    "rescale_guidance",
    True,
    "Rescale the shape gradient to --diffusion_guidance_strength x the denoiser's"
    " step norm. Off (--norescale_guidance): the raw gradient is applied, which is"
    " practically no shape guidance, and the strength has no effect.",
)
_GUIDANCE_GRADIENT = flags.DEFINE_enum(
    "guidance_gradient",
    "denoiser",
    ["denoiser", "x0hat"],
    "Shape gradient: pulled back through the denoiser to the noisy input (denoiser,"
    " J^T g; one extra denoiser forward + backward per step; the tested setting), or"
    " taken on the denoised prediction and applied directly (x0hat). The pose fit is"
    " the same in both.",
)
_GUIDANCE_CONFIG = flags.DEFINE_string(
    "guidance_config",
    None,
    "YAML file: which losses guide the sampler, their settings and schedules, the pose"
    " loss and optional guidance_steps. Default: cryoalpha/guiding/configs/default.yaml."
    " Format: the 'loss config (YAML)' comment in cryoalpha/guiding/setup_guidance.py.",
)
_GATHER_ATOMS = flags.DEFINE_boolean(
    "gather_atoms",
    True,
    "Give the guidance losses only the real atoms instead of all 24 slots per"
    " token (~3x faster, same result). Falls back automatically (with a message)"
    " for inputs or losses it can't handle.",
)
_GUIDANCE_VERBOSE = flags.DEFINE_boolean(
    "guidance_verbose",
    False,
    "Print the guidance diagnostics for every sample at every step ([JAC], step sizes,"
    " [POSE]; about 3000 lines per seed). Off: only [NAN] lines when the NaN guard fires.",
)


def _check_input(json_path, buckets) -> None:
  if not json_path or not os.path.isfile(json_path):
    return
  import json
  with open(json_path) as f:
    seqs = json.load(f).get("sequences", [])
  protein = seqs[0].get("protein") if len(seqs) == 1 else None
  if protein is None or (isinstance(protein.get("id"), list) and len(protein["id"]) > 1):
    print("[GUIDANCE] WARNING: the input is not a single protein chain. Guidance is only tested for one"
          " protein chain; the domain-integrity restraint does not support anything else.")
    return
  n = len(protein["sequence"])
  if buckets is None or [int(b) for b in buckets] != [n]:
    print(f"[GUIDANCE] WARNING: --buckets is {buckets}, the sequence has {n} residues."
          f" Guided runs were only tested with --buckets={n}.")


def _apply_guidance_from_flags() -> None:
  if _DENSITY_MAP_PATH.value is None:
    print("[GUIDANCE] No --density_map_path given, running unguided.")
    return

  guide = _GUIDE_DIFFUSION.value
  if not guide:
    print("[GUIDANCE] Diffusion guidance is disabled, running unguided.")
    return

  _check_input(flags.FLAGS.json_path, flags.FLAGS.buckets)

  config = load_guidance_config(_GUIDANCE_CONFIG.value)

  # Loaded outside jit.
  density = DensityMap(
      _DENSITY_MAP_PATH.value, cut_fraction=_MAP_CUT_FRACTION.value
  )

  if guide:
    total_loss = build_total_loss(
        density,
        config,
        json_path=flags.FLAGS.json_path,
    )

    n_atoms = None
    if _GATHER_ATOMS.value:
      n_atoms, reason = gather_atom_count(total_loss, flags.FLAGS.json_path)
      if n_atoms is None:
        print(f"[GUIDANCE] Atom gather off: {reason}")
    guided_head.set_guidance(
        total_loss=total_loss,
        guidance_strength=_DIFFUSION_GUIDANCE_STRENGTH.value,
        pose_loss=total_loss.pose_loss,   # config's pose_loss, unweighted: drives the rigid pose
        rescale=_RESCALE_GUIDANCE.value,
        n_atoms=n_atoms,
        gradient_mode=_GUIDANCE_GRADIENT.value,
        active_steps=config.get("guidance_steps"),
        random_augmentation=config["random_augmentation"],
        strength_schedule_t=build_strength_schedule_t(config),
        verbose=_GUIDANCE_VERBOSE.value,
    )
    print(
        f"[GUIDANCE] Diffusion guidance enabled"
        f" (strength={_DIFFUSION_GUIDANCE_STRENGTH.value},"
        f" rescale={_RESCALE_GUIDANCE.value},"
        f" gradient={_GUIDANCE_GRADIENT.value},"
        f" config={config['path']},"
        f" losses={[e['name'] for e in config['losses']]}, pose={config['pose_loss']},"
        f" guidance_steps={config.get('guidance_steps')},"
        f" gathered atoms={n_atoms},"
        f" random_augmentation={config['random_augmentation']},"
        f" strength_schedule={config.get('strength_schedule')})"
    )


# 4. Import the rest of the AlphaFold 3 runner.
import run_alphafold as af3_run_alphafold


def main(argv):
  _apply_guidance_from_flags()
  run_record.write_run_record(flags.FLAGS.output_dir, flags.FLAGS)
  af3_run_alphafold.main(argv)


if __name__ == "__main__":
  app.run(main)
