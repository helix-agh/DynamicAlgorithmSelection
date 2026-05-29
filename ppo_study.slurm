#!/bin/bash
#SBATCH --job-name=das2_ppo
#SBATCH --output=logs/ppo_%A_%a.out
#SBATCH --error=logs/ppo_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH -A plgrldas2026-gpu-a100
#SBATCH --array=0-9

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

DIMS=(2 3 5 10)

echo "Array job $SLURM_ARRAY_TASK_ID | SEED=$SEED | PORTFOLIO=${PORTFOLIO[*]}"

# 0-3: single-dim CV-LOIO
if [[ $SLURM_ARRAY_TASK_ID -ge 0 && $SLURM_ARRAY_TASK_ID -le 3 ]]; then
    DIM=${DIMS[$SLURM_ARRAY_TASK_ID]}
    echo "PPO | CV-LOIO | dim=$DIM"
    python cv.py ppo ${PORTFOLIO_STR}_PPO_LOIO_DIM${DIM}_SEED${SEED} \
        -p "${PORTFOLIO[@]}" -d $DIM --cv-mode LOIO --n-epochs 3 --seed $SEED -j 3

# 4-7: single-dim CV-LOPO
elif [[ $SLURM_ARRAY_TASK_ID -ge 4 && $SLURM_ARRAY_TASK_ID -le 7 ]]; then
    DIM=${DIMS[$((SLURM_ARRAY_TASK_ID - 4))]}
    echo "PPO | CV-LOPO | dim=$DIM"
    python cv.py ppo ${PORTFOLIO_STR}_PPO_LOPO_DIM${DIM}_SEED${SEED} \
        -p "${PORTFOLIO[@]}" -d $DIM --cv-mode LOPO --n-epochs 3 --seed $SEED -j 3

# 8: multi-dim CV-LOIO
elif [[ $SLURM_ARRAY_TASK_ID -eq 8 ]]; then
    echo "PPO | CV-LOIO | multi-dim"
    python cv.py ppo ${PORTFOLIO_STR}_PPO_LOIO_MULTI_SEED${SEED} \
        -p "${PORTFOLIO[@]}" -d 2 3 5 10 --cv-mode LOIO --n-epochs 3 --seed $SEED -j 3

# 9: multi-dim CV-LOPO
elif [[ $SLURM_ARRAY_TASK_ID -eq 9 ]]; then
    echo "PPO | CV-LOPO | multi-dim"
    python cv.py ppo ${PORTFOLIO_STR}_PPO_LOPO_MULTI_SEED${SEED} \
        -p "${PORTFOLIO[@]}" -d 2 3 5 10 --cv-mode LOPO --n-epochs 3 --seed $SEED -j 3
fi