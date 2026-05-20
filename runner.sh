#!/bin/bash

SEEDS=(12)

PORTFOLIOS=(
    "CPSO TDE NM"
)

echo "Starting job submissions..."

for SEED in "${SEEDS[@]}"; do
    for PORTFOLIO in "${PORTFOLIOS[@]}"; do

        echo "Submitting PPO study | SEED=$SEED | PORTFOLIO=$PORTFOLIO"
        sbatch ppo_study.slurm $SEED $PORTFOLIO
        sleep 1

        echo "Submitting RL-DAS study | SEED=$SEED | PORTFOLIO=$PORTFOLIO"
        sbatch rl_das_study.slurm $SEED $PORTFOLIO
        sleep 1

        echo "Submitting Exp-DAS study | SEED=$SEED | PORTFOLIO=$PORTFOLIO"
        sbatch exp_das_study.slurm $SEED $PORTFOLIO
        sleep 1

        echo "Submitting baselines | SEED=$SEED | PORTFOLIO=$PORTFOLIO"
        sbatch baselines.slurm $SEED $PORTFOLIO
        sleep 1

    done
done

echo "All jobs submitted!"