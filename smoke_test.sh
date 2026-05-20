#!/bin/bash
# Smoke tests — runs every agent type with tiny settings to catch import/wiring errors.
# Uses run_local.sh which already sets fe-multiplier=10, n-checkpoints=3, n-epochs=1.
#
# Usage:
#   ./smoke_test.sh              # run all tests
#   ./smoke_test.sh ppo rl-das   # run specific agents

set -uo pipefail

SEED=42
PASS=0
FAIL=0
FAILURES=()
LOG_DIR=$(mktemp -d)

run_smoke() {
    local label=$1
    local agent=$2
    local log="$LOG_DIR/${agent//-/_}.log"
    printf "%-25s ... " "$label"
    if bash run_local.sh $SEED "$agent" > "$log" 2>&1; then
        echo "PASS"
        ((PASS++)) || true
    else
        echo "FAIL  (log: $log)"
        ((FAIL++)) || true
        FAILURES+=("$label")
    fi
}

if [ "$#" -gt 0 ]; then
    AGENTS=("$@")
else
    AGENTS=(ppo ppo-cv rl-das rl-das-cv exp-das exp-das-cv baselines)
fi

for agent in "${AGENTS[@]}"; do
    case "$agent" in
        ppo)        run_smoke "ppo train"     ppo       ;;
        ppo-cv)     run_smoke "ppo cv"        ppo-cv    ;;
        rl-das)     run_smoke "rl-das train"  rl-das    ;;
        rl-das-cv)  run_smoke "rl-das cv"     rl-das-cv ;;
        exp-das)    run_smoke "exp-das train" exp-das   ;;
        exp-das-cv) run_smoke "exp-das cv"    exp-das-cv;;
        baselines)  run_smoke "baselines"     baselines ;;
        *)          echo "Unknown agent: $agent"; exit 1 ;;
    esac
done

echo ""
echo "Smoke tests: $PASS passed, $FAIL failed"
if [ "${#FAILURES[@]}" -gt 0 ]; then
    for f in "${FAILURES[@]}"; do
        echo "  FAILED: $f"
        echo "  --- log: $LOG_DIR/${f// /_}.log ---"
        tail -20 "$LOG_DIR/${f// /_}.log" 2>/dev/null || true
    done
    exit 1
fi