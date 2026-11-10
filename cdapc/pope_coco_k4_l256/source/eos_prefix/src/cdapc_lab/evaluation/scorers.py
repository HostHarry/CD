import ast
import re
from pathlib import Path
from .parsers import strict_parse
from dataclasses import asdict


def load_official_scorer(source_file):
    """Bind the exact official pure function without importing its training utilities."""
    source_file = Path(source_file)
    tree = ast.parse(source_file.read_text())
    nodes = [n for n in tree.body if
             (isinstance(n, ast.FunctionDef) and n.name == "judge_answer") or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ALPHA_MAP" for t in n.targets))]
    if len(nodes) != 2:
        raise ValueError("Pinned official scorer structure changed")
    namespace = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_file), "exec"), namespace)
    return namespace["judge_answer"]


def score_text(text, options, gold, official_scorer):
    parsed = strict_parse(text, options)
    return {"parsed": asdict(parsed),
            "strict_accuracy": int(parsed.status == "parsed" and parsed.prediction == gold),
            "official": int(official_scorer(text, list(options.values()), gold))}


def load_pope_official_rule(source_file):
    """Bind the answer-rewriting loop of the pinned RUCAIBox POPE evaluate.py."""
    source_file = Path(source_file)
    tree = ast.parse(source_file.read_text())
    loops = [n for n in tree.body if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == "answer"
             and isinstance(n.iter, ast.Name) and n.iter.id == "answers"
             and any(isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                             and t.value.id == "answer" and isinstance(t.slice, ast.Constant)
                             and t.slice.value == "answer"
                             for t in node.targets)
                     for node in ast.walk(n))]
    if len(loops) != 1:
        raise ValueError("Pinned official scorer structure changed")

    def rule(text):
        namespace = {"answers": [{"answer": text}]}
        exec(compile(ast.Module(body=loops, type_ignores=[]), str(source_file), "exec"), namespace)
        return namespace["answers"][0]["answer"]
    return rule


def score_pope_text(text, gold, rule, parser_id="strict_yes_no_v1"):
    from .parsers import strict_yes_no, reasoned_yes_no
    if parser_id == "pope_final_yes_no_v1":
        parsed = reasoned_yes_no(text)
        pred = rule(parsed.prediction) if parsed.prediction is not None else None
        return {"parsed": asdict(parsed), "official_prediction": pred,
                "official": int(pred is not None and pred == gold),
                "strict_accuracy": int(parsed.prediction is not None and parsed.prediction == gold),
                "raw_official_prediction": rule(text), "raw_official": int(rule(text) == gold)}
    if parser_id != "strict_yes_no_v1":
        raise ValueError(f"Unknown POPE parser: {parser_id}")
    parsed = strict_yes_no(text)
    official_prediction = rule(text)
    return {"parsed": asdict(parsed), "official_prediction": official_prediction,
            "official": int(official_prediction == gold),
            "strict_accuracy": int(parsed.status == "parsed" and parsed.prediction == gold)}


MMMU_FUNCTIONS = ("parse_multi_choice_response", "check_is_number", "normalize_str",
                  "extract_numbers", "parse_open_response", "eval_multi_choice", "eval_open")


class MmmuUnparsed(Exception):
    pass


class _SeededRandom:
    """random.choice bound to a per-sample seed, matching the official fallback path."""

    def __init__(self, seed):
        import random
        self._rng = random.Random(seed)

    def choice(self, seq):
        return self._rng.choice(seq)


class _StrictRandom:
    def choice(self, seq):
        raise MmmuUnparsed("random fallback is unparsed under the strict track")


def mmmu_random_for(sample_id):
    import hashlib
    seed = int(hashlib.sha256(f"mmmu_rule_v1:{sample_id}".encode()).hexdigest()[:16], 16)
    return _SeededRandom(seed)


def load_mmmu_official(eval_utils_file, data_utils_file):
    """Bind the pinned MMMU-Benchmark evaluator functions without importing the module.

    The returned scorer factory takes a per-call ``random`` object so that the
    official random-fallback is deterministic (official track) or raises
    (strict track)."""
    import numpy as np
    eval_file, data_file = Path(eval_utils_file), Path(data_utils_file)
    nodes = [n for n in ast.parse(eval_file.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in MMMU_FUNCTIONS]
    if {n.name for n in nodes} != set(MMMU_FUNCTIONS):
        raise ValueError("Pinned MMMU scorer structure changed")
    info = [n for n in ast.parse(data_file.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "get_multi_choice_info"]
    if len(info) != 1:
        raise ValueError("Pinned MMMU data_utils structure changed")
    module = ast.Module(body=nodes + info, type_ignores=[])

    def make_scorer(rng):
        namespace = {"re": re, "np": np, "random": rng}
        exec(compile(module, str(eval_file), "exec"), namespace)
        funcs = {name: namespace[name] for name in (*MMMU_FUNCTIONS, "get_multi_choice_info")}

        def score(text, options, gold, question_type):
            try:
                if question_type == "multiple-choice":
                    index2ans, all_choices = funcs["get_multi_choice_info"](list(options.values()))
                    parsed = funcs["parse_multi_choice_response"](text, all_choices, index2ans)
                    return {"parsed_pred": parsed,
                            "correct": bool(funcs["eval_multi_choice"](gold, parsed)),
                            "unparsed": False}
                parsed = funcs["parse_open_response"](text)
                return {"parsed_pred": parsed,
                        "correct": bool(funcs["eval_open"](gold, parsed)), "unparsed": False}
            except MmmuUnparsed:
                return {"parsed_pred": None, "correct": False, "unparsed": True}
        return score
    return make_scorer
