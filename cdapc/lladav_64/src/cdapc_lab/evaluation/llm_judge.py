"""DeepSeek LLM judge for MMMU-val, matching WINO's judge prompt and parse rule."""
import hashlib
import json
import re
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from cdapc_lab.io.artifacts import atomic_json, read_json, file_hash, safe_id, digest
from cdapc_lab.data.mmmu import mmmu_judge_question
from cdapc_lab.data.records import load_gold, load_manifest
from cdapc_lab.evaluation.normalize import normalize_output
from cdapc_lab.config import resolve_config

ROOT = Path(__file__).resolve().parents[3]
JUDGE_UTILS = ROOT / "upstream/lavida/eval/lmms_eval/tasks/mmmu/utils.py"
SYSTEM = "You are a helpful and precise assistant for checking the correctness of the answer."
ANSWER_RE = re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>")
BASE_URL = "https://api.deepseek.com"
RETRIES = 3


def load_judge_rules():
    import ast
    tree = ast.parse(JUDGE_UTILS.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "JUDGE_RULES" for t in n.targets)]
    if len(nodes) != 1:
        raise ValueError("Pinned JUDGE_RULES structure changed")
    return ast.literal_eval(nodes[0].value), file_hash(JUDGE_UTILS)


def judge_key():
    import os
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    path = Path("/root/.config/deepseek/api_key")
    return path.read_text().strip() if path.is_file() else None


def judge_spec(judge):
    rules, rules_hash = load_judge_rules()
    spec = {"model": judge, "base_url": BASE_URL, "temperature": 0.0, "max_tokens": 20,
            "thinking": {"type": "disabled"}, "system": SYSTEM, "rules_sha256": hashlib.sha256(rules.encode()).hexdigest(),
            "rules_source_sha256": rules_hash, "parse": "score=1 if '1' in content or '[1]' in content else 0",
            "strict_judge": "content.strip() in {'0','1'}",
            "answer_extraction": "<answer>\\s*([\\s\\S]*?)\\s*</answer>"}
    return spec, digest(spec)


def build_request(spec, question, gold, pred):
    rules, _ = load_judge_rules()
    return {"model": spec["model"], "messages": [
        {"role": "system", "content": spec["system"]},
        {"role": "user", "content": rules.format(question=question, answer=gold, pred=pred)}],
        "temperature": spec["temperature"], "max_tokens": spec["max_tokens"],
        "thinking": spec["thinking"]}


def _post(url, payload, key):
    request = urllib.request.Request(
        url + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def judge_prediction(spec, question, gold, pred, key):
    request = build_request(spec, question, gold, pred)
    request_sha = hashlib.sha256(
        json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    last_error = None
    for _ in range(RETRIES):
        try:
            response = _post(spec["base_url"], request, key)
            content = response["choices"][0]["message"]["content"]
            return {"request": request, "request_sha256": request_sha, "response": response,
                    "returned_model": response.get("model"), "usage": response.get("usage"),
                    "content": content,
                    "score": int("1" in content or "[1]" in content),
                    "strict_judge": content.strip() in {"0", "1"}}
        except Exception as error:
            last_error = repr(error)
            time.sleep(5)
    return {"request": request, "request_sha256": request_sha, "status": "judge_error",
            "error": last_error}


def mmmu_judge(run, judge="deepseek-flash"):
    run = Path(run)
    config = read_json(run / "resolved_config.json")
    protocol = config["protocol"]
    if protocol.get("task") != "mmmu":
        raise ValueError("mmmu-judge requires an MMMU run")
    spec, spec_hash = judge_spec(judge)
    out = run / "judge" / spec_hash
    out.mkdir(parents=True, exist_ok=True)
    manifest = {**spec, "judge_spec_hash": spec_hash}
    atomic_json(out / "judge_manifest.json", manifest)
    key = judge_key()
    if not key:
        return {"judge_spec_hash": spec_hash, "skipped": "no DEEPSEEK_API_KEY"}
    samples, manifest_info = load_manifest(protocol["data_root"], Path(protocol.get("sample_ids_file") or
                                        config["run"]["sample_ids_file"]), protocol["dataset_revision"], protocol["split"], "mmmu")
    gold = load_gold(protocol["data_root"], [s.sample_id for s in samples])
    records = {r["sample_id"]: r for r in read_json(Path(protocol["data_root"]) / "samples.json")}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["checkpoint"], local_files_only=True)
    done = scored = errors = 0
    for sample in samples:
        sid = sample.sample_id
        target = out / (safe_id(sid) + ".json")
        previous = read_json(target) if target.is_file() else None
        if previous is not None and previous.get("status") != "judge_error":
            done += 1
            continue
        prediction = read_json(run / "predictions" / (safe_id(sid) + ".json"))
        record = records[sid]
        pred = ""
        if "raw_response_ids" in prediction:
            pred = normalize_output(prediction["raw_response_ids"], tokenizer, prediction["eos_ids"])["text"]
        match = ANSWER_RE.search(pred)
        pred_judge = match.group(1) if match else pred
        result = judge_prediction(spec, mmmu_judge_question(record["question"], record["options"]),
                                  gold[sid].answer, pred_judge, key)
        result.update(sample_id=sid, pred=pred_judge)
        if previous is not None:
            result["previous_failures"] = previous.get("previous_failures", []) + [
                {k: v for k, v in previous.items() if k != "previous_failures"}]
        atomic_json(target, result)
        scored += int(result.get("score", 0) == 1)
        errors += int(result.get("status") == "judge_error")
        done += 1
    return {"judge_spec_hash": spec_hash, "judged": done, "scored_1": scored, "judge_errors": errors}
