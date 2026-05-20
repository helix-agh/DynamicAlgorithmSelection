#!/bin/bash
#SBATCH --job-name=das2_rldas
#SBATCH --output=logs/rldas_%A_%a.out
#SBATCH --error=logs/rldas_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH -A plgrldas2026-gpu-a100
#SBATCH --array=0-7

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

DIMS=(2 3 5 10)

echo "Array job $SLURM_ARRAY_TASK_ID | SEED=$SEED | PORTFOLIO=${PORTFOLIO[*]}"

# 0-3: CV-LOIO per dimension
if [[ $SLURM_ARRAY_TASK_ID -ge 0 && $SLURM_ARRAY_TASK_ID -le 3 ]]; then
    DIM=${DIMS[$SLURM_ARRAY_TASK_ID]}
    echo "RL-DAS | CV-LOIO | dim=$DIM"
    python cv.py rl-das ${PORTFOLIO_STR}_RLDAS_LOIO_DIM${DIM}_SEED${SEED} \
        -p "${PORTFOLIO[@]}" --dim $DIM --cv-mode LOIO --n-epochs 500 --seed $SEED

# 4-7: CV-LOPO per dimension
elif [[ $SLURM_ARRAY_TASK_ID -ge 4 && $SLURM_ARRAY_TASK_ID -le 7 ]]; then
    DIM=${DIMS[$((SLURM_ARRAY_TASK_ID - 4))]}
    echo "RL-DAS | CV-LOPO | dim=$DIM"
    python cv.py rl-das ${PORTFOLIO_STR}_RLDAS_LOPO_DIM${DIM}_SEED${SEED} \
        -p "${PORTFOLIO[@]}" --dim $DIM --cv-mode LOPO --n-epochs 500 --seed $SEED
fi