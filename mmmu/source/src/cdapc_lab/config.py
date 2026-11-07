from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import copy
import math
import yaml

VARIANTS = {
    "full": ("measured", "contrast", "contrast"),
    "gain_zero": ("zero", "contrast", "contrast"),
    "timing_only_crossed": ("measured", "contrast", "base"),
    "base_candidate_visual": ("measured", "base", "base"),
}


def fields_only(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"{label}: unknown/unused fields {sorted(unknown)}")
    return value


def positive_int(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")


@dataclass(frozen=True)
class Gate:
    enabled: bool = True
    threshold: float = 0.1

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("gate.enabled must be boolean")
        if not isinstance(self.threshold, (int, float)) or isinstance(self.threshold, bool):
            raise ValueError("gate.threshold must be numeric")
        if not math.isfinite(self.threshold) or not 0 < self.threshold <= 1:
            raise ValueError("gate.threshold must be in (0, 1]; disable with enabled=false")


@dataclass(frozen=True)
class FixedWindow:
    kind: str = "fixed"
    width: int = 16

    def __post_init__(self):
        if self.kind != "fixed":
            raise ValueError("Unknown fixed-window kind")
        positive_int(self.width, "window.width")


@dataclass(frozen=True)
class AdaptiveWindow:
    kind: str = "ccaw_v1"
    initial: int = 16
    minimum: int = 16
    maximum: int = 64
    ema_decay: float = .8
    expand_step: int = 4
    shrink_step: int = 8

    def __post_init__(self):
        if self.kind != "ccaw_v1":
            raise ValueError("Unknown window controller")
        for key in ("initial", "minimum", "maximum", "expand_step", "shrink_step"):
            positive_int(getattr(self, key), f"window.{key}")
        if not 1 <= self.minimum <= self.initial <= self.maximum:
            raise ValueError("Require minimum <= initial <= maximum")
        if type(self.ema_decay) not in (int, float) or not 0 <= self.ema_decay < 1:
            raise ValueError("Require 0 <= ema_decay < 1")


@dataclass(frozen=True)
class CDConfig:
    variant: str = "full"
    alpha: float = 0.5
    beta: float = 0.1
    u_gate: Gate = field(default_factory=Gate)
    r_gate: Gate = field(default_factory=lambda: Gate(threshold=0.9))
    max_commit_per_iteration: int = 16
    window: FixedWindow | AdaptiveWindow = field(default_factory=FixedWindow)
    branch_execution: str = "sequential"
    cache: str = "disabled"
    stop_rule: str = "fill_canvas"
    iteration_limit: int = 512
    budget_limited: bool = False

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown cdapc.variant: {self.variant}")
        for name in ("alpha", "beta"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite numeric")
        if self.alpha < 0 or not 0 < self.beta <= 1:
            raise ValueError("Require alpha >= 0 and 0 < beta <= 1")
        positive_int(self.max_commit_per_iteration, "K")
        positive_int(self.iteration_limit, "iteration_limit")
        minimum = self.window.width if self.window.kind == "fixed" else self.window.minimum
        if self.max_commit_per_iteration > minimum:
            raise ValueError("K must not exceed minimum W")
        if self.branch_execution not in ("sequential", "paired_batch"):
            raise ValueError("Unknown branch_execution")
        if self.cache != "disabled" or self.stop_rule not in ("fill_canvas", "eos_prefix_complete"):
            raise ValueError("Only cache=disabled with stop_rule fill_canvas/eos_prefix_complete supported")
        if type(self.budget_limited) is not bool:
            raise ValueError("budget_limited must be boolean")

    @property
    def binding(self):
        return dict(zip(("gain_used", "gate_target", "write_target"), VARIANTS[self.variant]))


def parse_cd(value):
    value = copy.deepcopy(value)
    fields_only(value, CDConfig.__dataclass_fields__.keys() | {"fallback"}, "cdapc")
    for name in ("u_gate", "r_gate"):
        if name in value:
            incoming = fields_only(value[name], {"enabled", "threshold"}, name)
            value[name] = Gate(**({"threshold": .9 if name == "r_gate" else .1} | incoming))
    if "window" in value:
        window_type = AdaptiveWindow if value["window"].get("kind") == "ccaw_v1" else FixedWindow
        value["window"] = window_type(**fields_only(value["window"], window_type.__dataclass_fields__, "window"))
    fallback = value.pop("fallback", {"position_rule": "normalized_gate_readiness", "token_rule": "base_top1"})
    if fallback != {"position_rule": "normalized_gate_readiness", "token_rule": "base_top1"}:
        raise ValueError("Only normalized_gate_readiness/base_top1 fallback supported")
    return CDConfig(**value)


def resolve_config(path):
    with Path(path).open() as handle:
        value = yaml.safe_load(handle)
    fields_only(value, {"schema_version", "run", "model", "runtime", "protocol", "method", "cdapc", "native", "recording"}, "config")
    if value.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    method = value.get("method")
    if method not in ("native", "cdapc"):
        raise ValueError(f"Unknown method: {method}")
    if method not in value or ({"native", "cdapc"} - {method}) & value.keys():
        raise ValueError("Require exactly the selected method's parameter block")
    sections = {
        "run": {"name": "experiment", "seed": 20261105, "replicate_id": 0, "sample_ids_file": None, "output_root": "runs", "fail_on_error": True},
        "model": {"backend": None, "checkpoint": None, "checkpoint_revision": None, "upstream_source_revision": None, "tokenizer_revision": None, "vision_component_revision": None, "dtype": "bfloat16", "attention_backend": "sdpa", "attention_semantics": "bidirectional", "upstream_root": None, "vision_checkpoint": None},
        "runtime": {"device": "cuda:0", "sample_batch_size": 1},
        "protocol": {"task": "m3cot", "dataset_revision": None, "data_root": None, "split": "test", "prompt_id": "m3cot_cot_final_v1", "canvas_length": 512, "normalizer_id": "first_eos_v1", "strict_parser_id": "strict_final_v1", "official_scorer_revision": None, "scorer_profile": "official_and_strict", "primary_metric": "official"},
        "recording": {"trace_level": "off", "trace_sample_ids_file": None, "snapshot_first_eligible_flip": True, "commit_log": False},
    }
    for name, defaults in sections.items():
        incoming = fields_only(value.get(name, {}), defaults, name)
        value[name] = defaults | incoming
        missing = [key for key, item in value[name].items() if item is None and not
                   (name == "recording" and key == "trace_sample_ids_file")]
        if missing:
            raise ValueError(f"{name}: missing required fields {missing}")
    p, m, r = value["protocol"], value["model"], value["recording"]
    positive_int(p["canvas_length"], "canvas_length")
    if type(value["run"]["seed"]) is not int or type(value["run"]["replicate_id"]) is not int:
        raise ValueError("seed/replicate_id must be integers")
    if type(value["run"]["fail_on_error"]) is not bool or type(r["snapshot_first_eligible_flip"]) is not bool \
            or type(r["commit_log"]) is not bool:
        raise ValueError("recording/run switches must be booleans")
    if m["backend"] not in ("mmada", "lavida"):
        raise ValueError("Unknown model backend")
    if m["dtype"] not in ("bfloat16", "float32") or m["attention_backend"] != "sdpa":
        raise ValueError("Only explicit bfloat16/float32 + SDPA supported")
    if m["attention_semantics"] != "bidirectional":
        raise ValueError("First protocol uses explicit bidirectional normal attention")
    if value["runtime"]["sample_batch_size"] != 1:
        raise ValueError("sample_batch_size must be 1")
    pairings = {"m3cot": {"prompt_id": "m3cot_cot_final_v1", "strict_parser_id": "strict_final_v1", "splits": ("train", "validation", "test")},
                "pope": {"prompt_id": "pope_yesno_v1", "strict_parser_id": "strict_yes_no_v1", "splits": ("test",)},
                "mmmu": {"prompt_id": "mmmu_wino_reasoning_v1", "strict_parser_id": "mmmu_rule_noguess_v1", "splits": ("validation",)}}
    if p["task"] not in pairings or p["split"] not in pairings[p["task"]]["splits"]:
        raise ValueError("Unsupported task/split")
    for key in ("prompt_id", "strict_parser_id"):
        if p[key] != pairings[p["task"]][key]:
            raise ValueError(f"Unsupported {key} for task {p['task']}: {p[key]}")
    for key, required in (("normalizer_id", "first_eos_v1"), ("scorer_profile", "official_and_strict")):
        if p[key] != required:
            raise ValueError(f"Unsupported {key}: {p[key]}")
    if p["primary_metric"] not in ("official", "strict_accuracy"):
        raise ValueError("Unsupported primary_metric")
    if r["trace_level"] not in ("off", "full", "sampled"):
        raise ValueError("Unknown trace_level")
    if (r["trace_level"] == "sampled") != (r["trace_sample_ids_file"] is not None):
        raise ValueError("trace_sample_ids_file is required only for sampled traces")
    if method == "cdapc":
        cd = parse_cd(value["cdapc"])
        if cd.branch_execution != "sequential":
            raise ValueError("paired_batch is verification-only: real BF16 decisions have not passed parity; use sequential")
        if cd.iteration_limit < p["canvas_length"] and not cd.budget_limited:
            raise ValueError("iteration_limit < L requires budget_limited=true")
        value["cdapc"] = asdict(cd)
        value["cdapc"]["fallback"] = {"position_rule": "normalized_gate_readiness", "token_rule": "base_top1"}
        value["derived"] = cd.binding
    else:
        n = {"steps": p["canvas_length"], "block_length": 128, "temperature": 0.0, "remasking": "low_confidence", "cfg_scale": 0.0, "cache": "disabled"}
        n.update(fields_only(value["native"], n, "native"))
        positive_int(n["steps"], "native.steps")
        positive_int(n["block_length"], "native.block_length")
        if p["canvas_length"] % n["block_length"] or n["steps"] % (p["canvas_length"] // n["block_length"]):
            raise ValueError("Native length/steps must divide into blocks")
        if n["cache"] != "disabled" or n["cfg_scale"] != 0.0:
            raise ValueError("Initial native protocol is no-cache, CFG=0")
        if type(n["temperature"]) not in (int, float) or not math.isfinite(n["temperature"]) or n["temperature"] < 0:
            raise ValueError("Invalid temperature")
        if n["remasking"] not in ("low_confidence", "random"):
            raise ValueError("Invalid remasking")
        value["native"] = n
        value["derived"] = {"execution_profile": "native_no_cache"}
    return value
