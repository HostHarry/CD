"""Finish the already-authorized VCD comparison once full CDAPC grading is ready."""
from collections import Counter
from datetime import datetime
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "cdapc-pope-coco-prefix-no-final"


def read(path):
    return json.loads(path.read_text())


def save(value):
    temp = ROOT / "reports/status.tmp"
    temp.write_text(json.dumps({**value, "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")}) + "\n")
    temp.replace(ROOT / "reports/status.json")


try:
    save({"status": "waiting_for_full_cdapc_judge", "controller_pid": os.getpid()})
    while True:
        generation = read(SOURCE / "reports/status.json")
        judge = read(SOURCE / "reports/judge/status.json")
        if generation["status"] == judge["status"] == "complete":
            break
        if generation["status"] == "interrupted_or_failed" or judge["status"] == "needs_attention":
            raise RuntimeError("Full CDAPC generation/scoring needs attention")
        time.sleep(15)
    records = read(SOURCE / "reports/prediction_inventory.json")
    rows = read(SOURCE / "manifests/rows.json")
    assert len(records) == 5127 and len(rows) == 9000
    assert set(records) == {r["input_key"] for r in rows}
    plan = {
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_workspace": str(SOURCE), "scope": "Full COCO matched comparison; preserve all terminal failures",
        "inputs": 5127, "rows": 9000, "images": 500,
        "generation_statuses": dict(Counter(v["status"] for v in records.values())),
        "split_rows": dict(Counter(r["split"] for r in rows)),
        "gold_labels": dict(Counter(r["label"] for r in rows)), "predictions": records,
        "preparation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT / "reports/plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (ROOT / "inputs/rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    save({"status": "verifying_full_matched_inputs", "controller_pid": os.getpid()})
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), "--prepare"], cwd=ROOT, check=True)
    os.execv(sys.executable, [sys.executable, str(ROOT / "scripts/evaluate.py")])
except BaseException as error:
    save({"status": "needs_attention", "error": repr(error), "controller_pid": os.getpid()})
    raise
