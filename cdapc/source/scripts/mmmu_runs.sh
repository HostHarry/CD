#!/usr/bin/env bash
# MMMU-val generation driver: for each method (native, cdapc_r90, cdapc_r50)
# run 3 shards on GPUs 0/1/2, resume once if incomplete, score, then next.
set -u
cd /root/autodl-tmp/cdapc-experiments
PY=.venv/bin/python
RUNS=reports/mmmu_runs.json
mkdir -p logs/mmmu reports
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
for METHOD in native cdapc_r90 cdapc_r50; do
  declare -a DIRS=() EVALS=()
  for K in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$K $PY -m cdapc_lab infer \
      --config configs/experiments/mmada_mmmu_${METHOD}_shard${K}.yaml \
      > logs/mmmu/${METHOD}-shard${K}.log 2>&1 &
  done
  wait
  for K in 0 1 2; do
    DIR=$(grep -o '"run": "[^"]*"' logs/mmmu/${METHOD}-shard${K}.log | tail -1 | cut -d'"' -f4)
    STATUS=$($PY -c "import json;print(json.load(open('$DIR/run_manifest.json'))['status'])" 2>/dev/null || echo missing)
    if [ "$STATUS" != "complete" ]; then
      CUDA_VISIBLE_DEVICES=$K $PY -m cdapc_lab infer \
        --config configs/experiments/mmada_mmmu_${METHOD}_shard${K}.yaml --resume-run "$DIR" \
        >> logs/mmmu/${METHOD}-shard${K}.log 2>&1
      DIR=$(grep -o '"run": "[^"]*"' logs/mmmu/${METHOD}-shard${K}.log | tail -1 | cut -d'"' -f4)
    fi
    DIRS[$K]=$DIR
    record "$METHOD" "shard$K" "$DIR"
  done
  for K in 0 1 2; do
    FP=$($PY -m cdapc_lab score --run "${DIRS[$K]}" | grep -o '"evaluation_fingerprint": "[^"]*"' | cut -d'"' -f4 | tail -1)
    EVALS[$K]=$FP
    record "$METHOD" "eval$K" "$FP"
  done
  record "$METHOD" evals "${EVALS[*]}"
done
# reports: cdapc_r90 vs native, cdapc_r50 vs native (official+strict; judge missing)
R90=$(grep -o '"run": "[^"]*"' logs/mmmu/cdapc_r90-shard*.log | tail -3 | cut -d'"' -f4 | tr '\n' ' ')
R50=$(grep -o '"run": "[^"]*"' logs/mmmu/cdapc_r50-shard*.log | tail -3 | cut -d'"' -f4 | tr '\n' ' ')
NAT=$(grep -o '"run": "[^"]*"' logs/mmmu/native-shard*.log | tail -3 | cut -d'"' -f4 | tr '\n' ' ')
$PY -m cdapc_lab mmmu-report --runs $R90 --native-runs $NAT \
  --output reports/mmmu_cdapc_r90.json > logs/mmmu/report-r90.log 2>&1
$PY -m cdapc_lab mmmu-report --runs $R50 --native-runs $NAT \
  --output reports/mmmu_cdapc_r50.json > logs/mmmu/report-r50.log 2>&1
# single judge probe
$PY - <<'PYEOF' > logs/mmmu/judge-probe.log 2>&1
import sys
sys.path.insert(0, "src")
from cdapc_lab.evaluation.llm_judge import judge_spec, build_request, judge_key, _post
key = judge_key()
if not key:
    print("NO_KEY")
else:
    spec, _ = judge_spec("deepseek-flash")
    req = build_request(spec, "Is water wet?", "yes", "yes")
    try:
        resp = _post(spec["base_url"], req, key)
        print("JUDGE_OK", resp.get("model"), resp["choices"][0]["message"]["content"])
    except Exception as e:
        print("JUDGE_FAIL", repr(e)[:200])
PYEOF
if grep -q JUDGE_OK logs/mmmu/judge-probe.log; then
  for METHOD in native cdapc_r90 cdapc_r50; do
    for K in 0 1 2; do
      DIR=$($PY -c "import json;print(json.load(open('$RUNS'))['$METHOD']['shard$K'])")
      $PY -m cdapc_lab mmmu-judge --run "$DIR" >> logs/mmmu/judge-${METHOD}-shard${K}.log 2>&1
    done
  done
  $PY -m cdapc_lab mmmu-report --runs $R90 --native-runs $NAT \
    --output reports/mmmu_cdapc_r90.json >> logs/mmmu/report-r90.log 2>&1
  $PY -m cdapc_lab mmmu-report --runs $R50 --native-runs $NAT \
    --output reports/mmmu_cdapc_r50.json >> logs/mmmu/report-r50.log 2>&1
fi
echo DONE
