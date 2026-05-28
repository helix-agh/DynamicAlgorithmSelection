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
#SBATCH --array=0-11

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

echo "Array job $SLURM_ARRAY_TASK_ID | SEED=$SEED | PORTFOLIO=${PORTFOLIO[*]}"

# 0: CV-LOIO multi-dim (2,5,10)
if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
    echo "Exp-DAS | CV-LOIO | multi-dim (2 5 10)"
    python cv.py exp-das ${PORTFOLIO_STR}_EXPDAS_LOIO_MULTI_SEED${SEED} \
        -p "${PORTFOLIO[@]}" --dims 2 5 10 --cv-mode LOIO --n-epochs 10 --seed $SEED --n-jobs 3

# 1: CV-LOPO multi-dim (2,5,10)
elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
    echo "Exp-DAS | CV-LOPO | multi-dim (2 5 10)"
    python cv.py exp-das ${PORTFOLIO_STR}_EXPDAS_LOPO_MULTI_SEED${SEED} \
        -p "${PORTFOLIO[@]}" --dims 2 5 10 --cv-mode LOPO --n-epochs 10 --seed $SEED --n-jobs 3

# 2: CV-LOIO multi-dim (2,3,5,10)
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then
    echo "Exp-DAS | CV-LOIO | multi-dim (2 3 5 10)"
    python cv.py exp-das ${PORTFOLIO_STR}_EXPDAS_LOIO_ALLDIM_SEED${SEED} \
        -p "${PORTFOLIO[@]}" --dims 2 3 5 10 --cv-mode LOIO --n-epochs 10 --seed $SEED --n-jobs 3

# 3: CV-LOPO multi-dim (2,3,5,10)
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]; then
    echo "Exp-DAS | CV-LOPO | multi-dim (2 3 5 10)"
    python cv.py exp-das ${PORTFOLIO_STR}_EXPDAS_LOPO_ALLDIM_SEED${SEED} \
        -p "${PORTFOLIO[@]}" --dims 2 3 5 10 --cv-mode LOPO --n-epochs 10 --seed $SEED --n-jobs 3

# 4-11: single-dimension runs; task ID encodes (dim_idx * 2 + cv_mode)
# dim_idx: 0→2, 1→3, 2→5, 3→10  |  cv_mode: 0→LOIO, 1→LOPO
else
    DIMS=(2 3 5 10)
    DIM_IDX=$(( (SLURM_ARRAY_TASK_ID - 4) / 2 ))
    CV_IDX=$(( (SLURM_ARRAY_TASK_ID - 4) % 2 ))
    DIM=${DIMS[$DIM_IDX]}
    CV_MODES=('LOIO' 'LOPO')
    CV_MODE=${CV_MODES[$CV_IDX]}
    echo "Exp-DAS | CV-${CV_MODE} | dim=${DIM}"
    python cv.py exp-das "${PORTFOLIO_STR}_EXPDAS_${CV_MODE}_${DIM}D_SEED${SEED}" \
        -p "${PORTFOLIO[@]}" --dims "${DIM}" --cv-mode "${CV_MODE}" --n-epochs 10 --seed "$SEED" --n-jobs 3
fi