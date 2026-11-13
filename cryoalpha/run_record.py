"""Write what defines a run to <output_dir>/run_config.json, before AF3 starts."""
import datetime
import json
import os
import socket
import subprocess

import numpy as np

from cryoalpha.af3_patches import diffusion_head
from cryoalpha.maps import DensityMap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALARS = (bool, int, float, str, type(None))


def _git(*args):
    out = subprocess.run(["git", "-C", PROJECT_ROOT, *args], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def describe(obj, depth=0):
    """JSON-able description of a loss / schedule / guidance object."""
    if isinstance(obj, SCALARS):
        return obj
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):          # numpy / jax array
        return f"array{tuple(obj.shape)}" if obj.shape else float(obj)
    if isinstance(obj, DensityMap):
        return {"type": "DensityMap", "file": obj.filepath, "cut_fraction": obj.cut_fraction}
    if isinstance(obj, (list, tuple)):
        return [describe(o, depth + 1) for o in obj]
    if callable(obj) and not hasattr(obj, "__dict__"):
        return getattr(obj, "__name__", type(obj).__name__)
    if depth > 4 or not hasattr(obj, "__dict__"):
        return type(obj).__name__
    fields = {}
    for k, v in vars(obj).items():
        if k.startswith("_"):
            continue
        d = describe(v, depth + 1)
        if d is not None or v is None:
            fields[k] = d
    return {"type": type(obj).__name__, **fields}


def _guidance():
    g = diffusion_head._GUIDANCE
    total = g["total_loss"]
    if total is None:
        return None
    return {
        "losses": [{"loss": describe(l), "schedule": describe(s)}
                   for l, s in zip(total.losses, total.schedulers)],
        "pose_loss": type(g["pose_loss"]).__name__,
        "guidance_strength": g["guidance_strength"],
        "rescale": g["rescale"],
        "n_atoms_gathered": g["n_atoms"],
        "gradient_mode": g["gradient_mode"],
        "active_steps": g["active_steps"],
        "random_augmentation": g.get("random_augmentation", False),
        "config": getattr(total, "config", None),   # the guidance YAML as loaded (incl. its path)
    }


def write_run_record(output_dir, flag_values):
    """flag_values: absl flags.FLAGS. Returns the path written."""
    os.makedirs(output_dir, exist_ok=True)
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--", "cryoalpha")
    record = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git": {"commit": commit, "uncommitted_changes_in_cryoalpha": bool(dirty)},
        "flags": {k: v for k, v in flag_values.flag_values_dict().items()
                  if isinstance(v, SCALARS) or (isinstance(v, list) and all(isinstance(x, SCALARS) for x in v))},
        "guidance": _guidance(),
        "sampler_constants": {k: v for k, v in vars(diffusion_head).items()
                              if k.isupper() and isinstance(v, (bool, int, float, str))},
    }
    try:
        gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
        record["gpu"] = gpu or record["gpu"]
    except FileNotFoundError:
        pass

    path = os.path.join(output_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    if dirty:
        with open(os.path.join(output_dir, "uncommitted.diff"), "w") as f:
            f.write(_git("diff", "HEAD", "--", "cryoalpha") or "")
        print(f"[RUN RECORD] WARNING: cryoalpha/ has uncommitted changes; saved as uncommitted.diff\n{dirty}")
    print(f"[RUN RECORD] wrote {path} (commit {commit[:7] if commit else 'unknown'})")
    return path
