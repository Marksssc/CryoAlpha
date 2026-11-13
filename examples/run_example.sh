#!/usr/bin/env bash
# Example: guide AlphaFold 3 towards the extended state 7MSL of A0A377JKY9 (88 residues) with a synthetic 4 A map.
# AF3 alone predicts the compact state (7F7P). Needs one GPU (~40 GB), about 1 min per seed after compilation.
#
#   export ALPHAFOLD_PREFIX=/path/to/alphafold3     # AF3 v3.0.4 checkout
#   export AF3_WEIGHTS=/path/to/af3_weights
#   bash examples/run_example.sh
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ALPHAFOLD_PREFIX:?set ALPHAFOLD_PREFIX to your AlphaFold 3 v3.0.4 checkout}"
: "${AF3_WEIGHTS:?set AF3_WEIGHTS to your AlphaFold 3 model parameters}"
EX=examples/A0A377JKY9
OUT=output/A0A377JKY9
N_RES=88

# 1. Unguided AF3 (one seed is enough): its model + PAE define the domains for the restraint.
python3 "${ALPHAFOLD_PREFIX}/run_alphafold.py" --json_path="${EX}/input_seed1.json" --model_dir="${AF3_WEIGHTS}" \
    --output_dir="${OUT}/unguided" --norun_data_pipeline --buckets=${N_RES}
REF=$(ls "${OUT}"/unguided/*/seed-1_sample-0/*_model.cif)
sed "s#/path/to/unguided_model.cif#${REF}#" configs/io.yaml > "${OUT}/io.yaml"

# 2. Guided AF3 (the input JSON has 40 seeds x 5 samples; edit modelSeeds for a quicker test).
python3 cryoalpha/run_alphafold.py --json_path="${EX}/input.json" --model_dir="${AF3_WEIGHTS}" \
    --output_dir="${OUT}/guided" --norun_data_pipeline --buckets=${N_RES} \
    --density_map_path="${EX}/7MSL_A_noisy.mrc" --guidance_config="${OUT}/io.yaml"

# 3. Pick the sample that fits the map best (CPU, no reference needed).
JAX_PLATFORMS=cpu python3 -m cryoalpha.rank_by_map "${OUT}/guided" "${EX}/7MSL_A_noisy.mrc" \
    --csv "${OUT}/ranking.csv" --write_best "${OUT}/best_placed.pdb"
