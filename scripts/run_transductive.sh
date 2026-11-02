#!/bin/bash
# Transductive GP regression benchmarks (100 seeds each): homoscedastic GP vs
# Forgetting to Improve (sequential removal, joint and aleatoric-only
# criterion) vs target-region sampling, relevance pursuit, warped GP and
# heteroscedastic GP.
#
# Usage (from the repository root):
#   bash scripts/run_transductive.sh             # all six benchmarks
#   bash scripts/run_transductive.sh twitter     # a subset
#   bash scripts/run_transductive.sh nn          # NN (last-layer Laplace) kernel variants
#   bash scripts/run_transductive.sh noise_ablation  # joint vs aleatoric-only criterion
#                                                    # (Forgetting to Improve only, twitter + sin_asymmetric)
#
# Results: forgetting_to_improve/forgetting_to_improve/results/comparison_<benchmark>_*.txt
# (one table per benchmark: mean ± std over seeds of RMSE, NLL, calibration, ...)
# plus the per-seed values in comparison_<benchmark>_*_per_seed.json.
# Figures: python -m forgetting_to_improve.forgetting_to_improve.plot_transductive {bar,noise_ablation}
set -e

CONFIG_DIR="forgetting_to_improve/forgetting_to_improve/configs"
declare -A CONFIGS=(
    [twitter]=twitter
    [yacht]=yacht
    [boston]=boston
    [casp]=casp
    [sin_symmetric]=statistical_comparison_sin_symmetric_lengthscale_increase
    [sin_asymmetric]=statistical_comparison_sin_asymmetric_lengthscale_increase
)
declare -A NN_CONFIGS=(
    [twitter]=statistical_twitter_jacobian
    [yacht]=statistical_yacht_jacobian
    [boston]=statistical_boston_jacobian
    [casp]=statistical_casp_jacobian
    [sin_symmetric]=statistical_sin_symmetric_jacobian
    [sin_asymmetric]=statistical_sin_asymmetric_jacobian
)
ORDER=(twitter yacht boston casp sin_symmetric sin_asymmetric)

if [ "$1" == "noise_ablation" ]; then
    for config in noise_ablation_twitter noise_ablation_sin_asymmetric; do
        echo "Running: ${config}.yaml  ($(date))"
        python -m forgetting_to_improve.forgetting_to_improve.main -c "${CONFIG_DIR}/${config}.yaml"
    done
    exit 0
fi

USE_NN=false
if [ "$1" == "nn" ]; then USE_NN=true; shift; fi
SELECTED=("$@")
[ ${#SELECTED[@]} -eq 0 ] && SELECTED=("${ORDER[@]}")

for name in "${SELECTED[@]}"; do
    if $USE_NN; then config="${NN_CONFIGS[$name]}"; else config="${CONFIGS[$name]}"; fi
    if [ -z "$config" ]; then echo "Unknown benchmark: $name (choose from ${ORDER[*]})"; exit 1; fi
    echo "=========================================="
    echo "Running: ${config}.yaml  ($(date))"
    echo "=========================================="
    python -m forgetting_to_improve.forgetting_to_improve.main -c "${CONFIG_DIR}/${config}.yaml"
done
