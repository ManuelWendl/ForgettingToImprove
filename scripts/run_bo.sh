#!/bin/bash
# Forgetful Bayesian optimization benchmarks (8 BoTorch test functions).
#
# Usage (from the repository root):
#   bash scripts/run_bo.sh            # GP experiments: UCB / LogEI, clean / corrupted
#   bash scripts/run_bo.sh nn         # NN (last-layer Laplace) kernel experiments
#   bash scripts/run_bo.sh branin     # only configs whose name contains "branin"
#
# Results:  forgetting_to_improve/forgetful_bayesian_optimization/results/<group>/
#           (group = ucb, ucb_noisy, lei, lei_noisy, nn_ucb, nn_ucb_noisy)
# Figures:  see README (plot_saved_results on each results/<group>/ folder)
set -e

CONFIG_DIR="forgetting_to_improve/forgetful_bayesian_optimization/configs"
BENCHMARKS=(ackley_2d ackley_6d augmented_rosenbrock branin griewank hartmann_6d holder_table levy)

CONFIGS=()
if [ "$1" == "nn" ]; then
    for b in "${BENCHMARKS[@]}"; do
        CONFIGS+=("${b}_jacobian" "${b}_jacobian_noisy")
    done
else
    for b in "${BENCHMARKS[@]}"; do
        CONFIGS+=("${b}" "${b}_noisy" "${b}_lei" "${b}_noisy_lei")
    done
    if [ -n "$1" ]; then
        FILTERED=()
        for c in "${CONFIGS[@]}"; do [[ "$c" == *"$1"* ]] && FILTERED+=("$c"); done
        CONFIGS=("${FILTERED[@]}")
    fi
fi

START_TIME=$(date +%s)
for config in "${CONFIGS[@]}"; do
    echo "=========================================="
    echo "Running: ${config}.yaml  ($(date))"
    echo "=========================================="
    python -m forgetting_to_improve.forgetful_bayesian_optimization.main -c "${CONFIG_DIR}/${config}.yaml"
done
DURATION=$(( $(date +%s) - START_TIME ))
echo "Done: ${#CONFIGS[@]} configs in $((DURATION / 3600))h $(((DURATION % 3600) / 60))m"
