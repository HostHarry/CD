#!/usr/bin/env bash
# Report POPE run progress: newest run dir per run.name with protocol.task == pope.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
.venv/bin/python - <<'PYEOF'
import json
from collections import Counter
from pathlib import Path

runs = Path("runs")
best = {}
if runs.is_dir():
    for run in sorted(runs.iterdir()):
        config_path = run / "resolved_config.json"
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            continue
        if config.get("protocol", {}).get("task") != "pope":
            continue
        name = config.get("run", {}).get("name", run.name)
        if name not in best or run.name > best[name].name:
            best[name] = run
for name in sorted(best):
    run = best[name]
    predictions = list((run / "predictions").glob("*.json")) if (run / "predictions").is_dir() else []
    counts = Counter()
    for path in predictions:
        try:
            counts[json.loads(path.read_text()).get("status", "?")] += 1
        except Exception:
            counts["unreadable"] += 1
    manifest_path = run / "run_manifest.json"
    status = json.loads(manifest_path.read_text()).get("status") if manifest_path.is_file() else "no manifest"
    print(f"{name}\t{run}\tpredictions={len(predictions)}\t{dict(counts)}\tstatus={status}")
PYEOF
