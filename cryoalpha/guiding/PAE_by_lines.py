"""Split a protein into domains from its PAE matrix (line-based: cuts where the PAE across a boundary is high).

Used by the domain-integrity restraint. To see the domains of an AF3 model before a guided run:

    python -m cryoalpha.guiding.PAE_by_lines <name>_confidences.json [--plot domains.png]
"""
import numpy as np

WINDOWS = [15, 6, 4]                # How large is the window for which you decide the size
MAD_MULTIPLIER = 1.5                # How much is the change for the motifs
MIN_DOMAIN_SIZE = 6                 # What is the minimal motif size


def boundary_score(dist: np.ndarray, w: int) -> np.ndarray:
    """score[i] = mean PAE of the block straddling the cut between local
    residues i and i+1, using up to w residues on each side."""
    m = dist.shape[0]
    score = np.empty(m - 1)
    for i in range(m - 1):
        lo = max(0, i - w + 1)
        hi = min(m, i + 1 + w)
        score[i] = dist[lo:i + 1, i + 1:hi].mean()
    return score


def find_boundaries(dist: np.ndarray, w: int) -> list[int]:
    m = dist.shape[0]
    if m - 1 <= 2:
        return []
    score = boundary_score(dist, w)
    median = np.median(score)
    mad = np.median(np.abs(score - median))
    threshold = median + MAD_MULTIPLIER * mad

    candidates = [
        i for i in range(1, len(score) - 1)
        if score[i] > threshold and score[i] >= score[i - 1] and score[i] >= score[i + 1]
        and (i + 1) >= MIN_DOMAIN_SIZE and (m - 1 - i) >= MIN_DOMAIN_SIZE
    ]
    candidates.sort(key=lambda i: -score[i])

    min_gap = max(2, w // 2)
    selected: list[int] = []
    for c in candidates:
        if all(abs(c - s) >= min_gap for s in selected):
            selected.append(c)
    return sorted(selected)


def segment(dist: np.ndarray, offset: int, level: int, leaves: list[tuple[int, int]],
            boundaries_by_level: dict[int, list[int]], windows: list[int]) -> None:
    m = dist.shape[0]
    if m < MIN_DOMAIN_SIZE * 2 or level >= len(windows):
        leaves.append((offset, offset + m - 1))
        return

    cuts = find_boundaries(dist, windows[level])
    if not cuts:
        segment(dist, offset, level + 1, leaves, boundaries_by_level, windows)
        return

    boundaries_by_level.setdefault(level, []).extend(offset + c for c in cuts)
    bounds = [-1] + cuts + [m - 1]
    for a, b in zip(bounds[:-1], bounds[1:]):
        piece = dist[a + 1:b + 1, a + 1:b + 1]
        segment(piece, offset + a + 1, level + 1, leaves, boundaries_by_level, windows)


def find_domains(pae_matrix: np.ndarray, subdomains: bool = False
                  ) -> tuple[int, list[tuple[int, int]]]:
    distances = (pae_matrix + pae_matrix.T) / 2.0
    active_windows = WINDOWS if subdomains else WINDOWS[:1]

    leaves: list[tuple[int, int]] = []
    boundaries_by_level: dict[int, list[int]] = {}
    segment(distances, 0, 0, leaves, boundaries_by_level, active_windows)
    leaves.sort()

    residue_ranges = [(start + 1, end + 1) for start, end in leaves]
    return len(residue_ranges), residue_ranges


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Domains from an AF3 PAE matrix")
    ap.add_argument("pae", help="AF3 *_confidences.json, or a .npy / .npz ('pae' key) matrix")
    ap.add_argument("--plot", help="write the PAE with the cuts drawn to this image file")
    args = ap.parse_args()
    if args.pae.endswith(".json"):
        PAE_matrix = np.array(json.load(open(args.pae))["pae"])
    else:
        PAE_matrix = np.load(args.pae)
        PAE_matrix = PAE_matrix["pae"] if hasattr(PAE_matrix, "files") else PAE_matrix
    distances = (PAE_matrix + PAE_matrix.T) / 2.0

    for label, sub in (("domains used by the restraint (coarse split)", False), ("finer split (not used)", True)):
        num_domains, residue_ranges = find_domains(PAE_matrix, subdomains=sub)
        print(f"{label}: {num_domains}")
        for start, end in residue_ranges:
            avg_pae = distances[start - 1:end, start - 1:end].mean()
            print(f"  residues {start}-{end} (len {end - start + 1}), avg internal PAE {avg_pae:.2f} A")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        leaves: list[tuple[int, int]] = []
        boundaries_by_level: dict[int, list[int]] = {}
        segment(distances, 0, 0, leaves, boundaries_by_level, WINDOWS)

        fig, ax = plt.subplots(figsize=(10, 8))

        cax = ax.imshow(distances, cmap='viridis_r', aspect='auto')
        fig.colorbar(cax, label='Predicted Aligned Error (Å)')

        coarse_cuts = boundaries_by_level.get(0, [])
        fine_cuts = [c for lvl, cuts in boundaries_by_level.items() if lvl > 0 for c in cuts]

        for c in fine_cuts:
            b = c + 0.5
            ax.axhline(b, color='white', linewidth=0.8, alpha=0.8, linestyle='--')
            ax.axvline(b, color='white', linewidth=0.8, alpha=0.8, linestyle='--')

        for c in coarse_cuts:
            b = c + 0.5
            ax.axhline(b, color='red', linewidth=2.5)
            ax.axvline(b, color='red', linewidth=2.5)

        ax.set_title(
            f"Line-based Domain Extraction (windows={WINDOWS}, MAD x{MAD_MULTIPLIER})\n"
            f"Red: coarsest-level cuts | White: finer cuts (recursed per piece, threshold recomputed each time)"
        )
        ax.set_xlabel("Residue Index")
        ax.set_ylabel("Residue Index")

        plt.tight_layout()
        fig.savefig(args.plot, dpi=200)
        print(f"-> {args.plot}")
