#!/bin/bash
# Local runner — no SLURM. Runs a small smoke-test configuration.
# Usage: ./run_local.sh [seed] [agent] [portfolio...]
#   agent: ppo | ppo-cv | rl-das | rl-das-cv | exp-das | exp-das-cv | baselines (default: ppo)

SEED=${1:-42}
AGENT=${2:-ppo}
shift 2 2>/dev/null || shift $#

if [ "$#" -lt 1 ]; then
    PORTFOLIO=('CPSO' 'NM' 'TDE')
else
    PORTFOLIO=("$@")
fi

PORTFOLIO_STR=$(IFS="_"; echo "${PORTFOLIO[*]}")

mkdir -p logs models results

echo "Local run | AGENT=$AGENT | SEED=$SEED | PORTFOLIO=${PORTFOLIO[*]}"

case "$AGENT" in
    ppo)
        python train.py ppo ${PORTFOLIO_STR}_PPO_LOCAL_SEED${SEED} \
            -p "${PORTFOLIO[@]}" -d 2 --n-epochs 1 --seed $SEED  --fe-multiplier 10  --n-checkpoints 3
        ;;
    ppo-cv)
        python cv.py ppo ${PORTFOLIO_STR}_PPO_CV_LOCAL_SEED${SEED} \
            -p "${PORTFOLIO[@]}" -d 2 --cv-mode LOIO --n-epochs 1 --seed $SEED  --fe-multiplier 10  --n-checkpoints 3
        ;;
    rl-das)
        python train.py rl-das NL_SHADE_RSP_MADDE_JDE21_RLDAS_LOCAL_SEED${SEED} \
            --dim 2 --n-epochs 1 --seed $SEED --fe-multiplier 10 --n-checkpoints 3
        ;;
    rl-das-cv)
        python cv.py rl-das NL_SHADE_RSP_MADDE_JDE21_RLDAS_CV_LOCAL_SEED${SEED} \
            --dim 2 --cv-mode LOIO --n-epochs 1 --seed $SEED --fe-multiplier 10 --n-checkpoints 3
        ;;
    exp-das)
        python train.py exp-das ${PORTFOLIO_STR}_EXPDAS_LOCAL_SEED${SEED} \
            -p "${PORTFOLIO[@]}" --dims 2 --n-epochs 1 --seed $SEED  --fe-multiplier 10  --n-checkpoints 3
        ;;
    exp-das-cv)
        python cv.py exp-das ${PORTFOLIO_STR}_EXPDAS_CV_LOCAL_SEED${SEED} \
            -p "${PORTFOLIO[@]}" --dims 2 --cv-mode LOIO --n-epochs 1 --seed $SEED --fe-multiplier 10  --n-checkpoints 3
        ;;
    baselines)
        python baselines.py ${PORTFOLIO_STR}_BASELINES_LOCAL_SEED${SEED} \
            -p "${PORTFOLIO[@]}" --agent all -d 2 --seed $SEED  --fe-multiplier 10 --n-checkpoints 3
        ;;
    *)
        echo "Unknown agent '$AGENT'. Use: ppo | ppo-cv | rl-das | rl-das-cv | exp-das | exp-das-cv | baselines"
        exit 1
        ;;
esac