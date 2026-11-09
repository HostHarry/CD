#!/usr/bin/env bash
# Sequential r-gate sweep on POPE: for each tau, run 3 shards on GPUs 0/1/2,
# resume once if incomplete, score, pope-report, then next tau.
set -u
cd /root/autodl-tmp/cdapc-experiments
PY=.venv/bin/python
RUNS=reports/pope_rgate_sweep_runs.json
mkdir -p logs/pope reports
[ -f "$RUNS" ] || echo '{}' > "$RUNS"
record() {
  $PY - "$RUNS" "$1" "$2" "$3" <<'PYEOF'
import json, sys
path, key, field, value = sys.argv[1:5]
data = json.load(open(path))
data.setdefault(key, {})[field] = value
json.dump(data, open(path, "w"), indent=2)
PYEOF
}
for TAU in 80 70 60 50; do
  KEY="r$TAU"; declare -a DIRS=() EVALS=()
  for K in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$K $PY -m cdapc_lab infer \
      --config configs/experiments/mmada_pope_yesno_r${TAU}_shard${K}.yaml \
      > logs/pope/r${TAU}-shard${K}.log 2>&1 &
  done
  wait
  for K in 0 1 2; do
    DIR=$(grep -o '"run": "[^"]*"' logs/pope/r${TAU}-shard${K}.log | tail -1 | cut -d'"' -f4)
    if [ -z "$DIR" ]; then
      DIR=$(ls -td runs/* 2>/dev/null | head -1)
    fi
    STATUS=$($PY -c "import json;print(json.load(open('$DIR/run_manifest.json'))['status'])" 2>/dev/null || echo missing)
    if [ "$STATUS" != "complete" ]; then
      CUDA_VISIBLE_DEVICES=$K $PY -m cdapc_lab infer \
        --config configs/experiments/mmada_pope_yesno_r${TAU}_shard${K}.yaml --resume-run "$DIR" \
        >> logs/pope/r${TAU}-shard${K}.log 2>&1
    fi
    DIRS[$K]=$DIR
    record "$KEY" "shard$K" "$DIR"
  done
  for K in 0 1 2; do
    FP=$($PY -m cdapc_lab score --run "${DIRS[$K]}" | grep -o '"evaluation_fingerprint": "[^"]*"' | cut -d'"' -f4 | tail -1)
    EVALS[$K]=$FP
    record "$KEY" "eval$K" "$FP"
  done
  $PY -m cdapc_lab pope-report --runs "${DIRS[@]}" --evaluations "${EVALS[@]}" \
    --native-export /root/autodl-tmp/HostHarry-CD \
    --output reports/pope_yesno_mmada_cdapc_L128_r${TAU}.json \
    > logs/pope/r${TAU}-report.log 2>&1
  record "$KEY" report "reports/pope_yesno_mmada_cdapc_L128_r${TAU}.json"
done
$PY scripts/pope_rgate_summary.py > logs/pope/rgate-summary.log 2>&1
echo DONE
