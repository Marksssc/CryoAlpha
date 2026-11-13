import copy
import dataclasses
import inspect
import json
import os

import numpy as np
import jax.numpy as jnp
import yaml

from cryoalpha.maps import DensityMap
from cryoalpha.guiding.losses import OverlapLoss, PairDistanceLoss, DomainIntegrityLoss
from cryoalpha.guiding import schedules
from cryoalpha.guiding.aggregator import TotalLoss
from cryoalpha.guiding.PAE_by_lines import find_domains

# True for every standard protein residue in AF3's DENSE_ATOM table
# (alphafold3/constants/atom_types.py) -- every entry starts (N, CA, C, O, ...).
CA_DENSE_ATOM_INDEX = 1


# Heavy atoms per standard residue, as AF3 places them.
HEAVY_ATOMS = dict(G=4, A=5, S=6, C=6, V=7, T=7, P=7, I=8, L=8, D=8, N=8, E=9, Q=9, M=8,
                   K=9, H=10, F=11, R=11, Y=12, W=14)
# Losses that can be given the real atoms alone instead of all 24 slots per token:
# density / p(r) see the atoms as a set, DomainIntegrityLoss carries its atom indices
# for both layouts.
GATHER_SAFE_LOSSES = (OverlapLoss, PairDistanceLoss, DomainIntegrityLoss)

def _load_pae(path):
    if path.endswith(".json"):                       # AF3 *_confidences.json
        with open(path) as f:
            return np.array(json.load(f)["pae"])
    pae = np.load(path)
    return pae["pae"] if hasattr(pae, "files") else pae


def gather_atom_count(total_loss: TotalLoss, json_path: str) -> tuple[int | None, str]:
    """Number of real atoms for the sampler's atom gather, or None (with the reason).

    Only for a single standard protein chain (one copy, no modifications), with
    only gather-safe losses.
    """
    used = list(total_loss.losses) + [getattr(total_loss, "pose_loss", None)]
    bad = [type(l).__name__ for l in used if l is not None and not isinstance(l, GATHER_SAFE_LOSSES)]
    if bad:
        return None, f"losses that need the 24-slot layout: {bad}"
    with open(json_path) as f:
        seqs = json.load(f).get("sequences", [])
    if len(seqs) != 1 or "protein" not in seqs[0]:
        return None, "input is not a single protein chain"
    protein = seqs[0]["protein"]
    if isinstance(protein.get("id"), list) and len(protein["id"]) > 1:
        return None, "more than one copy of the chain"
    if protein.get("modifications"):
        return None, "modified residues"
    unknown = set(protein["sequence"]) - set(HEAVY_ATOMS)
    if unknown:
        return None, f"non-standard residues {sorted(unknown)}"
    return sum(HEAVY_ATOMS[a] for a in protein["sequence"]) + 1, ""


# ------------------------------------------------------------------ loss config (YAML)
#
# Which losses guide the sampler, with what settings and when, comes from a YAML file
# (--guidance_config; default: configs/default.yaml = p(r) early + overlap late, no restraint;
# the recommended setup with the domain restraint is configs/io.yaml at the repository root):
#
#   pose_loss: overlap            # name of the entry that drives the rigid pose fit, or none
#                                 # (no pose fit; only orientation-free losses: pr, domain_integrity)
#   guidance_steps: [100, 155]    # optional: guide only steps 100 <= step < 155 (no pose fit outside)
#   random_augmentation: true     # optional, only with pose_loss none: AF3's own random rotation +
#                                 # translation at every step (as unguided AF3). Default false.
#   strength_schedule: {type: StepSchedule, start_val: 1.0, end_val: 0.5, threshold: 0.7, input_type: t}
#                                 # optional: multiplier on --diffusion_guidance_strength over the steps
#                                 # (input_type t only; e.g. half strength from step 140). Default: none.
#   losses:
#     - name: overlap             # a key of LOSSES below
#       params: {resolution: 4.0} # constructor arguments of that loss (checked)
#       schedule: {type: PartSchedule, val: 1.0, peak_start: 0.5, peak_end: 0.8, input_type: blur}
#       shape: true               # optional; false = only for the pose fit, not in the shape loss
#
# Schedules are classes of guiding/schedules.py; input_type "t" = step / 200, "blur" = the blur.
# Entries are summed in file order.

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
MAX_ATOMS_PER_TOKEN = 24


def _overlap(params, ctx):
    return OverlapLoss(density_map=ctx["density"], **params)


def _pr(params, ctx):
    return PairDistanceLoss(density_map=ctx["density"], **params)

def _read_ca(cif_path, n_res):
    """CA coordinates (n_res, 3) of an AF3 model CIF, by label_seq_id (1-based)."""
    cols, ca = [], np.full((n_res, 3), np.nan)
    for line in open(cif_path):
        if line.startswith("_atom_site."):
            cols.append(line.strip().split(".", 1)[1])
        elif line.startswith("ATOM") and cols:
            d = dict(zip(cols, line.split()))
            if d["label_atom_id"] == "CA":
                ca[int(d["label_seq_id"]) - 1] = [float(d["Cartn_x"]), float(d["Cartn_y"]), float(d["Cartn_z"])]
    if np.isnan(ca).any():
        raise ValueError(f"{cif_path}: CA missing for residues {np.nonzero(np.isnan(ca[:, 0]))[0] + 1}")
    return ca


def _ref_pae_path(ref_model):
    """AF3 writes <name>_seed-S_sample-K_confidences.json next to each model CIF."""
    path = ref_model.replace("_model.cif", "_confidences.json")
    if path == ref_model or not os.path.exists(path):
        raise ValueError(f"no AF3 confidences JSON next to {ref_model} (expected {path})")
    return path


def _domain_integrity(params, ctx):
    """CA elastic network per motif vs a reference model set by the PAE, more uncertain motifs get more interior movement."""
    p = dict(params)
    with open(ctx["json_path"]) as f:
        seq = json.load(f)["sequences"][0]["protein"]["sequence"]
    n = len(seq)
    ca = _read_ca(p["ref_model"], n)
    pae = _load_pae(_ref_pae_path(p["ref_model"]))
    pae = 0.5 * (pae + pae.T)
    if pae.shape != (n, n):
        raise ValueError(f"PAE {pae.shape} does not match the sequence ({n} residues)")
    _, doms = find_domains(pae)
    pairs, d_ref, tol = [], [], []
    for a, b in doms:
        r = np.arange(a - 1, b)                                   # 0-based residues of this piece
        d = np.linalg.norm(ca[r][:, None] - ca[r][None], axis=-1)
        i, j = np.nonzero(np.triu(d < p.get("cutoff", 15.0), k=1))
        pairs.append(np.stack([r[i], r[j]], 1)); d_ref.append(d[i, j]); tol.append(pae[r[i], r[j]])
    pairs, d_ref, tol = np.concatenate(pairs), np.concatenate(d_ref), np.concatenate(tol)
    counts = np.array([HEAVY_ATOMS[a] for a in seq]); counts[-1] += 1      # OXT on the last residue
    ca_gathered = np.concatenate([[0], np.cumsum(counts)[:-1]]) + 1
    ca_slots = np.arange(n) * MAX_ATOMS_PER_TOKEN + CA_DENSE_ATOM_INDEX
    per = ", ".join(f"{a}-{b} (PAE {pae[a-1:b, a-1:b].mean():.1f})" for a, b in doms)
    print(f"[DOMAIN INTEGRITY] pieces {per}; {len(pairs)} CA pairs, tolerance {tol.min():.1f}-{tol.max():.1f} Å")
    return DomainIntegrityLoss(ca_gathered=jnp.asarray(ca_gathered[pairs]), ca_slots=jnp.asarray(ca_slots[pairs]),
                               n_gathered=int(counts.sum()), d_ref=jnp.asarray(d_ref, jnp.float32),
                               tol=jnp.asarray(tol, jnp.float32))

# name -> (builder, loss class whose constructor arguments `params` may set, extra params)
LOSSES = {
    "overlap": (_overlap, OverlapLoss, ()),             # Gaussian density correlation
    "pr": (_pr, PairDistanceLoss, ()),                  # pair-distance function p(r)
    "pr_fine": (_pr, PairDistanceLoss, ()),             # same loss, second entry (e.g. fine extra_blur late)
    "domain_integrity_late": (_domain_integrity, DomainIntegrityLoss,   # second entry (own window and weight)
                              ("ref_model", "cutoff")),
    "domain_integrity": (_domain_integrity, DomainIntegrityLoss,
                        ("ref_model", "cutoff")),
}
# Set by the builder, not by the config.
_FIXED = {"density_map", "self", "ca_gathered", "ca_slots", "n_gathered", "d_ref", "tol"}


def _check_keys(where, given, allowed):
    unknown = sorted(set(given) - set(allowed))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _build_schedule(spec, where):
    spec = dict(spec)
    kind = spec.pop("type", None)
    cls = getattr(schedules, kind, None) if kind else None
    if cls is None or not dataclasses.is_dataclass(cls):
        names = sorted(n for n, c in vars(schedules).items() if dataclasses.is_dataclass(c))
        raise ValueError(f"{where}: unknown schedule type {kind!r}; one of {names}")
    _check_keys(f"{where} ({kind})", spec, [f.name for f in dataclasses.fields(cls)])
    return cls(**spec)


def load_guidance_config(path: str | None = None) -> dict:
    """Read and check a guidance config (YAML). None -> configs/default.yaml."""
    path = path or DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    _check_keys(path, cfg, ["pose_loss", "guidance_steps", "losses", "random_augmentation",
                            "strength_schedule"])
    entries = cfg.get("losses") or []
    if not entries:
        raise ValueError(f"{path}: no losses")
    names = []
    for i, e in enumerate(entries):
        where = f"{path}: losses[{i}]"
        _check_keys(where, e, ["name", "params", "schedule", "shape"])
        if e.get("name") not in LOSSES:
            raise ValueError(f"{where}: unknown loss {e.get('name')!r}; one of {sorted(LOSSES)}")
        builder, cls, extra = LOSSES[e["name"]]
        allowed = [p for p in inspect.signature(cls).parameters if p not in _FIXED] + list(extra)
        _check_keys(f"{where} ({e['name']}) params", e.get("params") or {}, allowed)
        if e.get("shape", True) and "schedule" not in e:
            raise ValueError(f"{where}: needs a schedule (or shape: false for a pose-only loss)")
        if "schedule" in e:
            _build_schedule(e["schedule"], f"{where} schedule")       # check now, not on the GPU
        names.append(e["name"])
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: a loss name appears twice; each entry needs its own name")
    pose = cfg.get("pose_loss", "overlap")
    if pose in (None, "none"):
        pose = None                     # no pose fit: only for orientation-free losses (pr, domain_integrity)
        orient = [n for n in names if n == "overlap"]
        if orient:
            raise ValueError(f"{path}: pose_loss none, but {orient} need the structure posed in the map")
    elif pose not in names:
        raise ValueError(f"{path}: pose_loss {pose!r} is not one of the losses {names} (or none)")
    steps = cfg.get("guidance_steps")
    if steps is not None:
        if len(steps) != 2 or not 0 <= int(steps[0]) < int(steps[1]):
            raise ValueError(f"{path}: guidance_steps must be [START, END] with 0 <= START < END")
        cfg["guidance_steps"] = [int(steps[0]), int(steps[1])]
    aug = cfg.get("random_augmentation", False)
    if not isinstance(aug, bool):
        raise ValueError(f"{path}: random_augmentation must be true or false")
    if aug and pose is not None:
        # The pose fit keeps a map-frame offset across steps; a random rotation every step would undo it.
        raise ValueError(f"{path}: random_augmentation needs pose_loss none")
    cfg["random_augmentation"] = aug
    ss = cfg.get("strength_schedule")
    if ss is not None:
        if ss.get("input_type") != "t":
            raise ValueError(f"{path}: strength_schedule needs input_type: t (it runs over the steps)")
        _build_schedule(ss, f"{path}: strength_schedule")          # check now, not on the GPU
    cfg["pose_loss"] = pose
    cfg["path"] = os.path.abspath(path)
    return cfg


def build_total_loss(
    density: DensityMap,
    config: dict,
    json_path: str | None = None,
) -> TotalLoss:
    """The diffusion-guidance TotalLoss for a density map, as described by `config`
    (load_guidance_config). The pose loss is attached as `.pose_loss`, the config as `.config`.
    """
    ctx = dict(density=density, json_path=json_path)
    losses, schedulers, pose_loss = [], [], None
    for e in config["losses"]:
        builder = LOSSES[e["name"]][0]
        loss = builder(dict(e.get("params") or {}), ctx)
        if e["name"] == config["pose_loss"]:
            pose_loss = loss
        if e.get("shape", True):
            losses.append(loss)
            schedulers.append(_build_schedule(e["schedule"], e["name"]))
    total = TotalLoss(losses=losses, schedulers=schedulers)
    total.pose_loss = pose_loss
    total.config = copy.deepcopy(config)
    return total

def build_strength_schedule_t(config: dict):
    """The guidance YAML's optional strength_schedule (over t = step / n_steps), or None."""
    ss = config.get("strength_schedule")
    return None if ss is None else _build_schedule(ss, "strength_schedule")
