#!/bin/bash
#SBATCH --job-name=das2_expdas
#SBATCH --output=logs/expdas_%A_%a.out
#SBATCH --error=logs/expdas_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH -A plgrldas2026-gpu-a100
#SBATCH --array=0-3

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

echo "Array job $SLURM_ARRAY_TASK_ID | SEED=$SEED | PORTFOLIO=${PORTFOLIO[*]}"

# 0: CV-LOIO multi-dim (2,5,10)
if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
    echo "Exp-DAS | CV-LOIO | multi-dim (2 5 10)"
    python cv.py exp-das ${PORTFOLIO_STR}_EXPDAS_LOIO_MULTI_SEED${SEED} \
        -p "${PORTFOLIO[@]}" --dims 2 5 10 --cv-mode LOIO --n-epochs 10 --seed $SEED

# 1: CV-LOPO multi-dim (2,5,10)
elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
    echo "Exp-DAS | CV-LOPO | multi-dim (2 5 10)"
    python cv.py exp-das ${PORTFOLIO_STR}_EXPDAS_LOPO_MULTI_SEED${SEED} \
        -p "${PORTFOLIO[@]}" --dims 2 5 10 --cv-mode LOPO --n-epochs 10 --seed $SEED

# 2: CV-LOIO multi-dim (2,3,5,10)
elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then
    echo "Exp-DAS | CV-LOIO | multi-dim (2 3 5 10)"
    python cv.py exp-das ${PORTFOLIO_STR}_EXPDAS_LOIO_ALLDIM_SEED${SEED} \
        -p "${PORTFOLIO[@]}" --dims 2 3 5 10 --cv-mode LOIO --n-epochs 10 --seed $SEED

# 3: CV-LOPO multi-dim (2,3,5,10)
elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]; then
    echo "Exp-DAS | CV-LOPO | multi-dim (2 3 5 10)"
    python cv.py exp-das ${PORTFOLIO_STR}_EXPDAS_LOPO_ALLDIM_SEED${SEED} \
        -p "${PORTFOLIO[@]}" --dims 2 3 5 10 --cv-mode LOPO --n-epochs 10 --seed $SEED
fi