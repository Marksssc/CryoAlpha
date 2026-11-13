"""Rank the samples of an AF3 run by their fit to a cryo-EM map, without a reference structure. (CPU)

Guidance gives many samples of which only a few are correct; the one that fits the map best is the answer.
Every sample is placed in the map using the map only (no reference, no sequence):
  1. 24 starts: the sample's principal axes onto the map's principal axes (thresholded, 3 A smoothed map),
     4 proper sign combinations x 6 turns about the map's long axis;
  2. local rigid fit (Nelder-Mead) on the mean smoothed density at the atoms from every start;
  3. the best 3 starts refined on the 1.5 A smoothed map; the highest map CC is kept.
Map CC = correlation of the map with a Gaussian density of the sample's heavy atoms (width set by --resolution).

Writes a CSV (seed, sample, map_cc, mean pLDDT, path) sorted by map CC and, with --write_best, the best sample
placed in the map frame as a PDB-format file (heavy atoms; overlays the map in ChimeraX).

    python -m cryoalpha.rank_by_map AF3_OUTPUT_DIR MAP.mrc [--resolution 4] [--workers 4] [--csv ranking.csv]
                                [--write_best best_placed.pdb]
"""
import argparse
import csv
import glob
import os
import re
from multiprocessing import Pool

import numpy as np
from scipy import ndimage, optimize
from scipy.spatial.transform import Rotation

from cryoalpha.maps import DensityMap

SEARCH_BLUR = 3.0    # A: smoothing of the map for the 24-start search
FIT_BLUR = 1.5       # A: smoothing for the final refinement
MAP_CUT = 0.2        # x max of the smoothed map: voxels that define the map's principal axes
N_REFINE = 3
RE_SAMPLE = re.compile(r"seed-(\d+)_sample-(\d+)")

_M = {}              # map state, set in _setup() before the worker pool forks


def read_cif(path):
    """Heavy atoms of an AF3 model: xyz, atom name, residue name, residue number, chain, element, B-factor."""
    cols, rows = [], []
    for line in open(path):
        if line.startswith("_atom_site."):
            cols.append(line.strip().split(".", 1)[1])
        elif line.startswith(("ATOM", "HETATM")) and cols:
            d = dict(zip(cols, line.split()))
            if d["type_symbol"] != "H":
                rows.append(d)
    xyz = np.array([[float(d["Cartn_x"]), float(d["Cartn_y"]), float(d["Cartn_z"])] for d in rows])
    return xyz, rows


def find_samples(run_dir):
    """{(seed, sample): model.cif} under an AF3 output folder."""
    found = {}
    for d in sorted(glob.glob(os.path.join(run_dir, "**", "seed-*_sample-*"), recursive=True)):
        m = RE_SAMPLE.fullmatch(os.path.basename(d))
        cif = glob.glob(os.path.join(d, "*_model.cif"))
        if m and cif and os.path.isdir(d):
            key = (int(m.group(1)), int(m.group(2)))
            if key in found:
                raise SystemExit(f"seed {key[0]} sample {key[1]} found twice; pass the folder of a single AF3 run")
            found[key] = cif[0]
    if not found:
        raise SystemExit(f"no seed-*_sample-*/*_model.cif under {run_dir}")
    return found


def _axes(xyz, w=None):
    c = np.average(xyz, axis=0, weights=w)
    _, v = np.linalg.eigh(np.cov((xyz - c).T, aweights=w))
    return c, v[:, ::-1]                                       # columns: long axis first


def _setup(map_path, resolution):
    dm = DensityMap(map_path)
    vs = float(dm.voxel_size[0])
    search = ndimage.gaussian_filter(dm.padded_data, SEARCH_BLUR / vs)
    mask = search > MAP_CUT * search.max()
    vz = np.array(np.nonzero(mask)).T[:, ::-1]                 # (x, y, z) padded index
    c, ax = _axes((vz - dm.pad_voxels) * dm.voxel_size + dm.origin, search[mask])
    ax[:, 2] *= np.sign(np.linalg.det(ax))                     # right-handed frame, or every start is a mirror image
    _M.update(dm=dm, search=search, smooth=ndimage.gaussian_filter(dm.padded_data, FIT_BLUR / vs),
              c=c, ax=ax, sigma=resolution / (np.pi * np.sqrt(2.0)) / vs)


def _index(xyz):
    dm = _M["dm"]
    return ((xyz - dm.origin) / dm.voxel_size + dm.pad_voxels)[:, ::-1].T


def map_cc(xyz):
    dm = _M["dm"]
    sim = np.zeros_like(dm.padded_data)
    idx = np.round(_index(xyz)).astype(int)
    idx = idx[:, np.all((idx >= 0) & (idx < np.array(sim.shape)[:, None]), axis=0)]
    np.add.at(sim, tuple(idx), 1.0)
    sim = ndimage.gaussian_filter(sim, _M["sigma"])
    return float(np.corrcoef(sim.ravel(), dm.padded_data.ravel())[0, 1])


def _local_fit(start, grid, step_rot, step_trans):
    c = _M["c"]
    place = lambda p: (start - c) @ Rotation.from_rotvec(p[:3]).as_matrix().T + c + p[3:]
    density = lambda x: ndimage.map_coordinates(grid, _index(x), order=1).mean()
    simplex = np.vstack([np.zeros(6), np.diag([step_rot] * 3 + [step_trans] * 3)])
    r = optimize.minimize(lambda p: -density(place(p)), np.zeros(6), method="Nelder-Mead",
                          options=dict(initial_simplex=simplex, xatol=1e-3, fatol=1e-7, maxiter=3000))
    return r.fun, place(r.x)


def place_in_map(xyz):
    """Best rigid placement of xyz (heavy atoms) in the map, using the map only. Returns (placed xyz, map CC)."""
    c, ax = _axes(xyz)
    x0 = xyz - c
    found = []
    for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        A = ax * np.array([sx, sy, sx * sy * np.linalg.det(ax)])   # proper rotations only
        for k in range(6):
            R0 = Rotation.from_rotvec(_M["ax"][:, 0] * k * np.pi / 3).as_matrix() @ _M["ax"] @ A.T
            found.append(_local_fit(x0 @ R0.T + _M["c"], _M["search"], 0.15, 2.0))
    found.sort(key=lambda f: f[0])
    best = None
    for _, start in found[:N_REFINE]:
        _, placed = _local_fit(start, _M["smooth"], 0.05, 0.5)
        cc = map_cc(placed)
        if best is None or cc > best[1]:
            best = (placed, cc)
    return best


def _score(item):
    (seed, sample), path = item
    xyz, rows = read_cif(path)
    _, cc = place_in_map(xyz)
    plddt = np.mean([float(d["B_iso_or_equiv"]) for d in rows if d["label_atom_id"] == "CA"])
    return dict(seed=seed, sample=sample, map_cc=round(cc, 4), plddt=round(float(plddt), 2), path=path)


def write_pdb(path, xyz, rows):
    with open(path, "w") as f:
        for i, (p, d) in enumerate(zip(xyz, rows), 1):
            name = d["label_atom_id"]
            f.write(f"ATOM  {i % 100000:5d} {name if len(name) == 4 else ' ' + name:<4} {d['label_comp_id']:>3} "
                    f"{d.get('auth_asym_id', d['label_asym_id'])[:1]}{int(d['label_seq_id']):4d}    "
                    f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00{float(d['B_iso_or_equiv']):6.2f}"
                    f"          {d['type_symbol']:>2}\n")
        f.write("END\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", help="AF3 output folder of one run (holds seed-*_sample-* folders)")
    ap.add_argument("map", help="the cryo-EM map (.mrc / .map)")
    ap.add_argument("--resolution", type=float, default=4.0, help="map resolution in A (width of the model density)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--csv", default="ranking.csv")
    ap.add_argument("--write_best", help="write the best sample, placed in the map, to this PDB file")
    args = ap.parse_args()

    _setup(args.map, args.resolution)
    items = sorted(find_samples(args.run_dir).items())
    print(f"{len(items)} samples, map {args.map}")
    with Pool(args.workers) as pool:
        rows = sorted(pool.map(_score, items), key=lambda r: -r["map_cc"])
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows[:5]:
        print(f"  seed {r['seed']:>3} sample {r['sample']}  map CC {r['map_cc']:.3f}  pLDDT {r['plddt']:.1f}")
    print(f"-> {args.csv} (best first)")
    if args.write_best:
        xyz, atoms = read_cif(rows[0]["path"])
        placed, _ = place_in_map(xyz)
        write_pdb(args.write_best, placed, atoms)
        print(f"-> {args.write_best} (best sample, placed in the map frame)")


if __name__ == "__main__":
    main()
