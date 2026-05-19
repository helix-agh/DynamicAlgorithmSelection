#!/bin/bash
#SBATCH --job-name=das2_baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH -A plgrldas2026-gpu-a100

# Args: SEED [PORTFOLIO...]
SEED=${1:-42}

if [ "$#" -lt 2 ]; then
    PORTFOLIO=('SPSO' 'IPSO' 'SPSOL')
else
    PORTFOLIO=("${@:2}")
fi

PORTFOLIO_STR=$(IFS="_"; echo "${PORTFOLIO[*]}")

ENV_PATH="$SCRATCH/DynamicAlgorithmSelection2/.venv/bin/activate"
source "$ENV_PATH"
mkdir -p logs

echo "Baselines | SEED=$SEED | PORTFOLIO=${PORTFOLIO[*]}"

python baselines.py ${PORTFOLIO_STR}_BASELINES_SEED${SEED} \
    -p "${PORTFOLIO[@]}" --agent all --seed $SEED