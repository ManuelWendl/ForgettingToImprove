#!/bin/bash
# Forgetful RL on cartpole balance: MBPO with a linear-kernel GP dynamics model,
# with (forgetful_rl=true) and without (GP-MBPO) replay-buffer forgetting.
# All settings come from forgetting_to_improve/forgetful_rl/config (main.yaml,
# algorithm/mbpo.yaml, dynamics_model/gp_model.yaml, overrides/mbpo_cartpole.yaml).
#
# Usage (from the repository root):
#   bash scripts/run_rl_cartpole.sh            # learning curves: 5 seeds x {forgetful, GP-MBPO}
#   bash scripts/run_rl_cartpole.sh removal    # one forgetful run (seed 1) that also stores
#                                              # which samples were removed at every filtering step
# Extra Hydra overrides are passed through, e.g. device=cpu or wandb.mode=disabled.
#
# Runs are written to exp/mbpo/<experiment>/cartpole_continuous/<date>/<time>/
# (results.jsonl, .hydra/config.yaml, data_removal/*.npz). Plot with
#   python -m forgetting_to_improve.forgetful_rl.plotting.plot_cartpole_balance --runs-dir exp/mbpo/cartpole_balance
#   python -m forgetting_to_improve.forgetful_rl.plotting.plot_removal_visualization --runs-dir exp/mbpo/cartpole_balance_removal
set -e

MODE="${1:-curves}"
[ "$MODE" == "removal" ] || [ "$MODE" == "curves" ] && shift || true

if [ "$MODE" == "removal" ]; then
    python -m forgetting_to_improve.forgetful_rl.main \
        seed=1 overrides.forgetful_rl=true overrides.store_data_removal=true \
        experiment=cartpole_balance_removal "$@"
    exit 0
fi

for seed in 0 1 2 3 4; do
    for forgetful in true false; do
        echo "=== seed ${seed}, forgetful_rl=${forgetful} ($(date))"
        python -m forgetting_to_improve.forgetful_rl.main \
            seed=${seed} overrides.forgetful_rl=${forgetful} \
            experiment=cartpole_balance "$@"
    done
done
