# CryoAlpha: cryo-EM map-guided sampling for AlphaFold 3

AlphaFold 3 usually predicts one conformation of a protein, the one it prefers. CryoAlpha steers AlphaFold 3's
diffusion sampler towards the conformation in a cryo-EM density map, without retraining and without the need for a docked
starting model. It is a patch on top of an unmodified AlphaFold 3 v3.0.4 installation and is untested on other versions.

How it works (default setup, `configs/io.yaml`):

1. **Early sampling** (steps 0–127 of 200): guidance by the map's pair distance distribution p(r). This needs
   no alignment to the map. A **domain-integrity restraint** keeps the domains (loose term, read motif) of an unguided AF3 prediction
   intact (domains and tolerances from its PAE).
2. **Late sampling** (steps 140–153): guidance by the overlap with the map density. A rigid pose fit runs on
   the predicted structure throughout.
3. The loss gradient is pulled back through the denoiser (the default gradient mode) and rescaled to
   10% of the denoiser's step.
4. Only a fraction of samples reach the target, so many samples are drawn and the one that fits the map best
   is selected (`cryoalpha.rank_by_map`, which needs no reference structure).

On synthetic 4 Å maps this recovers large hinge openings that unguided AlphaFold 3 and the existing guidance
methods CryoBoltz and CryoACE miss. See *Limitations* below before using it.

## Installation

1. **AlphaFold 3 v3.0.4** (Python 3.12, `jax==0.10.2`). Install it from
   <https://github.com/google-deepmind/alphafold3> (tag `v3.0.4`), following DeepMind's instructions, and
   request the model parameters from Google. The parameters are subject to
   [their own terms of use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md) and
   are not part of this repository.
2. **This package**, in the same Python environment:
   ```bash
   git clone https://github.com/Marksssc/CryoAlpha.git
   cd CryoAlpha
   pip install -e .
   ```
3. Point it at your AlphaFold 3 checkout (the folder that contains DeepMind's `run_alphafold.py`):
   ```bash
   export ALPHAFOLD_PREFIX=/path/to/alphafold3
   ```

The patch replaces AlphaFold 3's diffusion head at startup and is written for v3.0.4 only. On another version
it prints a warning and the results may then be wrong.

## Quick start

`examples/` holds a complete test case: A0A377JKY9 (88 residues), for which AlphaFold 3 predicts the compact
state 7F7P, with a synthetic 4 Å map of the extended state 7MSL.

```bash
export ALPHAFOLD_PREFIX=/path/to/alphafold3
export AF3_WEIGHTS=/path/to/af3_weights
bash examples/run_example.sh
```

The script runs three steps:

1. **Unguided AF3, one seed.** Its model and PAE define the domains for the restraint.
2. **Guided AF3.** 40 seeds × 5 samples; reduce `modelSeeds` in the input JSON for a quick test.
3. **Ranking by map fit.** Writes `output/A0A377JKY9/ranking.csv` and the best sample placed in the map
   (`best_placed.pdb`).

On the benchmark about 5% of the guided samples of this case reach TM-score > 0.9 to 7MSL, and the map-CC pick
does (TM-score ~0.97). On other proteins the hit rate may vary depending on the difference between the conformations.

## Usage on your own data

```bash
python3 cryoalpha/run_alphafold.py \
    --json_path=input.json --model_dir=$AF3_WEIGHTS --output_dir=out --norun_data_pipeline \
    --buckets=<number of residues> \
    --density_map_path=map.mrc --guidance_config=my_io.yaml

JAX_PLATFORMS=cpu python3 -m cryoalpha.rank_by_map out map.mrc --resolution 4 --write_best best_placed.pdb
```

The flag defaults are the method as tested: gradient through the denoiser, rescaled to 10% of the denoiser's
step (`--guidance_gradient=denoiser --rescale_guidance --diffusion_guidance_strength=0.10`). Always pass a
`--guidance_config` based on `configs/io.yaml`: the built-in default config has no domain restraint.
`--guidance_gradient=x0hat` applies the gradient directly instead of through the denoiser, is faster but does not always produce good results.

- **Input JSON.** An AlphaFold 3 input with MSA and templates already filled in: run AF3's data pipeline once,
  then use `--norun_data_pipeline`. Use many seeds; 100–200 samples per target is what the method was tested with.
- **`--buckets`** must equal the sequence length: a single protein chain is assumed, token *i* = residue *i* + 1.
- **`my_io.yaml`.** A copy of `configs/io.yaml` with `ref_model` set to an unguided AF3 model of the same input.
  Its `*_confidences.json` must sit next to it.
- **Map.** An `.mrc` map that covers the predicted sequence. The map should be masked around the protein, and
  `resolution` in the YAML should be set to the map's resolution.

Guided models are written centred at the origin, not in the map's frame: use `rank_by_map --write_best` to get
the selected model placed in the map. To see which domains the restraint will use, run
`python -m cryoalpha.guiding.PAE_by_lines <unguided>_confidences.json`.

All other AlphaFold 3 flags work as usual. Run `python3 cryoalpha/run_alphafold.py --helpfull` for the guidance
flags. Every run writes `run_config.json` (all flags and the guidance setup) into the output folder.

The log is quiet by default (only `[NAN]` lines, when a sample's guided step was non-finite and the unguided
step was used instead). `--guidance_verbose` prints the pose fit and step sizes of every sample at every step.

### Guidance configuration (YAML)

| key | meaning |
|---|---|
| `pose_loss` | loss used for the rigid pose fit (`overlap`), or `none` for orientation-free losses only |
| `guidance_steps` | `[start, end]` diffusion steps with guidance; outside it AF3 runs unguided |
| `losses` | list of losses, each with `name`, `params` and a `schedule` (its weight over blur or step) |
| `random_augmentation` | keep AF3's random rotation per step (only with `pose_loss: none`) |
| `strength_schedule` | optional multiplier on `--diffusion_guidance_strength` over the steps |

Losses: `overlap` (density overlap), `pr` (pair distance distribution), `domain_integrity` (restraint).
The full format is documented in the comment block "loss config (YAML)" in
`cryoalpha/guiding/setup_guidance.py`.

`configs/l_late.yaml` extends the late overlap phase to step 175. This gives sharper picks and more correct
samples on moderate changes, but more clashes.

## Limitations

- **Single protein chain, standard residues.** Multimers, ligands and modified residues are not supported.
- **Low fraction of correct samples in hard cases** (typically 2–10%), so many samples and map-based selection are needed.
- **Can lower accuracy where AF3 is already right.** If the map shows the state AF3 already predicts, guidance
  can make it worse. Compare the best map fit of guided and unguided samples.
- **The map must cover the predicted sequence.** Residues without density are pushed into the map and the
  structure breaks; trim the sequence to what the map resolves.
- **Memory and compute.** About 40–80 GB of GPU memory, depending on the map box. Each guided sample is 10–30×
  slower than unguided AF3.
- **Not bit-reproducible on GPU.** Compare settings over many seeds, not per seed.

## Acknowledgements

CryoAlpha was developed during an internship in the Lindahl group at SciLifeLab, under the daily supervision of
Samuel Eriksson Lidbrink and the overall supervision of Dr. Rebecca Howard.

## Licence

Apache License 2.0 (see `LICENSE`). `cryoalpha/af3_patches/diffusion_head.py` is a modified copy of AlphaFold
3's `diffusion_head.py` (Copyright 2024 DeepMind Technologies Limited, Apache 2.0); its header states the
changes. AlphaFold 3 model parameters are not included and are subject to Google's terms of use.
