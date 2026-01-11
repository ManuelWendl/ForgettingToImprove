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
CONFIG_DIR="forgetting_to_improve/forgetfull_bayesian_optimization/configs"

# Array of config files to run
CONFIGS=(
    "ackley_2d.yaml",
    "ackley_6d.yaml",
    "branin.yaml",
    "griewank.yaml",
    "hartmann_6d.yaml",
    "holder_table.yaml",
    "levy.yaml",
    "rosenbrock.yaml",
)

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
        
        python -m forgetting_to_improve.forgetfull_bayesian_optimization.main -c "$CONFIG_PATH"
        
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
