"""Verify this export using only Python's standard library; no model or network."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def metrics(pairs):
    counts = Counter()
    for pred, gold in pairs:
        cell = ("TP" if pred == "yes" and gold == "yes" else "FP" if pred == "yes" else
                "TN" if pred == "no" and gold == "no" else "FN" if gold == "yes" else "unresolved")
        counts[cell] += 1
    tp, fp, tn, fn = (counts[k] for k in ("TP", "FP", "TN", "FN"))
    return {"accuracy": (tp + tn) / sum(counts.values()),
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
            "precision": tp / (tp + fp) if tp + fp else 0,
            "recall": tp / (tp + fn) if tp + fn else 0,
            "yes_ratio": (tp + fp) / sum(counts.values())}


def main():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert sha((ROOT / name).read_bytes()) == expected, name
    package = load(ROOT / "manifest.json")
    for path, provenance in package["copied_file_provenance"].items():
        assert sha((ROOT / path).read_bytes()) == provenance["sha256"], path
    rows = load(ROOT / "rows.json")
    expected_ids = {r["input_key"] for r in rows}
    assert len(rows) == len({r["row_id"] for r in rows}) == 9000
    assert len(expected_ids) == 5127
    assert Counter(r["split"] for r in rows) == {"random": 3000, "popular": 3000, "adversarial": 3000}
    for name, condition in package["conditions"].items():
        folder = ROOT / name
        records = load(folder / "records_index.json")
        assert records.keys() == expected_ids
        answers_list = [json.loads(line) for line in (folder / "answers.jsonl").read_text().splitlines()]
        answers = {r["input_key"]: r for r in answers_list}
        assert answers.keys() == expected_ids and len(answers_list) == len(answers)
        official, judge = load(folder / "official.json"), load(folder / "judge.json")
        source = ROOT / "source" / condition["source_snapshot"] / "src/cdapc_lab"
        paths = set()
        for relative in package["source_dependencies"]:
            p = source / relative
            paths.update(p.rglob("*.py") if p.is_dir() else [p])
        code_hash = sha(canonical({str(p.relative_to(source)): sha(p.read_bytes()) for p in sorted(paths)}))
        assert code_hash == condition["generation_source_hash"]
        expected_members = defaultdict(dict)
        for sid, entry in records.items():
            for kind in ("prediction", "score"):
                expected_members[entry["archive"]][entry[kind]["member"]] = (sid, kind, entry[kind]["sha256"])
        statuses, seen = Counter(), set()
        run_statuses, run_seen = defaultdict(Counter), defaultdict(set)
        for archive, members in expected_members.items():
            with tarfile.open(ROOT / archive, "r|gz") as stream:
                for member in stream:
                    assert member.isfile() and member.name in members
                    sid, kind, expected = members[member.name]
                    assert (sid, kind) not in seen
                    seen.add((sid, kind))
                    raw = stream.extractfile(member).read()
                    assert sha(raw) == expected
                    value = json.loads(raw)
                    assert value["sample_id"] == sid
                    answer = answers[sid]
                    if kind == "prediction":
                        manifest = load(folder / "runs" / records[sid]["run"] / "run_manifest.json")
                        assert manifest["status"] == (
                            "incomplete" if manifest["sample_status_counts"].get("budget_exhausted") else "complete")
                        assert set(manifest["sample_status_counts"]) <= {"success", "budget_exhausted"}
                        assert manifest["generation_source_hash"] == code_hash
                        assert value["generation_fingerprint"] == manifest["generation_fingerprint"]
                        assert value["status"] == answer["generation_status"]
                        state = [value["mask_id"]] * len(value["raw_response_ids"])
                        for event in value["commit_log"]:
                            pos = event["position"]
                            assert state[pos] == value["mask_id"]
                            state[pos] = event["written_id"]
                        assert state == value["raw_response_ids"], (name, sid, "trace reconstruction")
                        assert value["cost"]["model_forward_calls"] == 2 * value["decode_iterations"]
                        eos = next((i for i, token in enumerate(state) if token in value["eos_ids"]), None)
                        assert eos == value["first_eos_position"]
                        if value["status"] == "success" and eos is not None:
                            assert value["mask_id"] not in state[:eos + 1]
                        statuses[value["status"]] += 1
                        run_statuses[records[sid]["run"]][value["status"]] += 1
                        run_seen[records[sid]["run"]].add(sid)
                    else:
                        assert value["normalized"]["text"] == answer["text"]
                        selected = answer["text"].rsplit("</think>", 1)[-1].strip()
                        assert selected == answer["answer_region"]
                        words = selected.split(".")[0].replace(",", "").split(" ")
                        pred = "no" if any(w in words for w in ("No", "not", "no")) else "yes"
                        assert pred == answer["official_prediction"]
                        effective = pred if answer["generation_status"] == "success" else None
                        assert effective == answer["effective_prediction"] == official["per_input"][sid]["effective_prediction"]
                    j = judge["per_input"][sid]
                    assert j["judge_prediction"] == answer["judge_prediction"]
                    assert j["judge_status"] == answer["judge_status"]
        assert len(seen) == 2 * len(expected_ids) and dict(statuses) == condition["generation_statuses"]
        assert set(run_seen) == set(condition["runs"])
        for run, ids in run_seen.items():
            manifest = load(folder / "runs" / run / "run_manifest.json")
            dataset = load(folder / "runs" / run / "dataset_manifest.json")
            assert ids == {sample["sample_id"] for sample in dataset["samples"]}
            assert len(ids) == manifest["sample_count"]
            assert dict(run_statuses[run]) == manifest["sample_status_counts"]
        for field, saved in (("effective_prediction", official), ("judge_prediction", judge)):
            split_metrics = []
            for split in ("random", "popular", "adversarial"):
                got = metrics((answers[r["input_key"]][field], r["label"]) for r in rows if r["split"] == split)
                for key, number in got.items():
                    assert abs(number - saved["splits"][split][key]) < 1e-12, (name, field, split, key)
                split_metrics.append(got)
            for key in split_metrics[0]:
                assert abs(sum(m[key] for m in split_metrics) / 3 - saved["macro"][key]) < 1e-12
        print(name, "5127 payloads, 9000 rows, original file hashes, source fingerprints, traces and both scores OK")
    print("All package checks passed")


if __name__ == "__main__":
    main()
