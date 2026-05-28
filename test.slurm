#!/bin/bash
#SBATCH --job-name=das2_expdas
#SBATCH --output=logs/expdas_%A_%a.out
#SBATCH --error=logs/expdas_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH -A plgrldas2026-gpu-a100

# Args: SEED [PORTFOLIO...]
SEED=${1:-42}

if [ "$#" -lt 2 ]; then
    PORTFOLIO=('NM' 'TDE' 'SPSOL')
else
    PORTFOLIO=("${@:2}")
fi

PORTFOLIO_STR=$(IFS="_"; echo "${PORTFOLIO[*]}")

ENV_PATH="$SCRATCH/DynamicAlgorithmSelection2/.venv/bin/activate"
source "$ENV_PATH"
mkdir -p logs

echo "Exp-DAS | CV-LOIO | dim=2"
python cv.py exp-das "NM_TDE_SPSOL_EXPDAS_LOIO_2D_SEED42" \
    -p "${PORTFOLIO[@]}" --dims 2 --cv-mode LOIO --n-epochs 10 --seed "$SEED" --n-jobs 3