#!/bin/bash

# Script to run all Bayesian Optimization experiments sequentially
# Usage: ./bo.bash

set -e  # Exit on error

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Define the config directory
CONFIG_DIR="forgetting_to_improve/forgetful_bayesian_optimization/configs"

# Array of config files to run
# CONFIGS=(
    # "ackley_2d.yaml"
    # "ackley_6d.yaml"
    # "augmented_rosenbrock.yaml"
    # "branin.yaml"
    # "griewank.yaml"
    # "hartmann_6d.yaml"
    # "holder_table.yaml"
    # "levy.yaml"
    # "ackley_2d_noisy.yaml"
    # "ackley_6d_noisy.yaml"
    # "augmented_rosenbrock_noisy.yaml"
    # "branin_noisy.yaml"
    # "griewank_noisy.yaml"
    # "hartmann_6d_noisy.yaml"
    # "holder_table_noisy.yaml"
    # "levy_noisy.yaml"
    # "ackley_2d_lei.yaml"
    # "ackley_6d_lei.yaml"
    # "augmented_rosenbrock_lei.yaml"
    # "branin_lei.yaml"
    # "griewank_lei.yaml"
    # "hartmann_6d_lei.yaml"
    # "holder_table_lei.yaml"
    # "levy_lei.yaml"
    # "ackley_2d_noisy_lei.yaml"
    # "ackley_6d_noisy_lei.yaml"
    # "augmented_rosenbrock_noisy_lei.yaml"
    # "branin_noisy_lei.yaml"
    # "griewank_noisy_lei.yaml"
    # "hartmann_6d_noisy_lei.yaml"
    # "holder_table_noisy_lei.yaml"
    # "levy_noisy_lei.yaml"
    # )


CONFIGS=(
    # "ackley_2d_jacobian.yaml"
    # "ackley_6d_jacobian.yaml"
    # "augmented_rosenbrock_jacobian.yaml"
    # "branin_jacobian.yaml"
    # "griewank_jacobian.yaml"
    # "hartmann_6d_jacobian.yaml"
    # "holder_table_jacobian.yaml"
    # "levy_jacobian.yaml"
    # "ackley_2d_jacobian_noisy.yaml"
    # "ackley_6d_jacobian_noisy.yaml"
    # "augmented_rosenbrock_jacobian_noisy.yaml"
    # "branin_jacobian_noisy.yaml"
    # "griewank_jacobian_noisy.yaml"
    # "hartmann_6d_jacobian_noisy.yaml"
    # "holder_table_jacobian_noisy.yaml"
    # "levy_jacobian_noisy.yaml"
    "ackley_2d_identity_jacobian.yaml"
)

# CONFIGS=(
    # "ackley_2d_noisy_lei.yaml"
    # "ackley_6d_noisy_lei.yaml"
#     "augmented_rosenbrock_noisy_lei.yaml"
#     "branin_noisy_lei.yaml"
#     "griewank_noisy_lei.yaml"
#     "hartmann_6d_noisy_lei.yaml"
#     "holder_table_noisy_lei.yaml"
#     "levy_noisy_lei.yaml"
# )

# Start time
START_TIME=$(date +%s)
echo "=========================================="
echo "Starting BO experiments at $(date)"
echo "=========================================="
echo ""

# Run each config
for config in "${CONFIGS[@]}"; do
    CONFIG_PATH="${CONFIG_DIR}/${config}"
    
    if [ -f "$CONFIG_PATH" ]; then
        echo "=========================================="
        echo "Running: $config"
        echo "Time: $(date)"
        echo "=========================================="
        
        python -m forgetting_to_improve.forgetful_bayesian_optimization.main -c "$CONFIG_PATH"
        
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "ERROR: $config failed with exit code $EXIT_CODE"
            exit $EXIT_CODE
        fi
        
        echo ""
        echo "Completed: $config"
        echo ""
    else
        echo "WARNING: Config file not found: $CONFIG_PATH"
    fi
done

# End time and summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "=========================================="
echo "All experiments completed at $(date)"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="
