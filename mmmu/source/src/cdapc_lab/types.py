from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any
import torch


@dataclass(frozen=True)
class Sample:
    sample_id: str
    question: str
    options: dict[str, str]
    image_paths: tuple[str, ...]
    task_id: str = "m3cot"
    group_id: str | None = None
    public_context: str | None = None


@dataclass(frozen=True)
class GoldRecord:
    sample_id: str
    answer: str
    metric_metadata: dict = field(default_factory=dict)


@dataclass
class ComputeCost:
    model_forward_calls: int = 0
    denoiser_branch_evaluations: int = 0
    full_sequence_branch_evaluations: int = 0
    prefill_calls: int = 0
    logical_query_positions: int = 0

    def __add__(self, other):
        return ComputeCost(**{f.name: getattr(self, f.name) + getattr(other, f.name)
                              for f in fields(self)})


@dataclass(frozen=True)
class PreparedInput:
    sample_id: str
    prompt_ids: torch.Tensor
    visual_keys: torch.Tensor
    allowed_text_vocab: torch.Tensor
    mask_id: int
    eos_ids: tuple[int, ...]
    pad_id: int | None
    canvas_length: int
    input_fingerprint: str
    model_inputs: dict[str, Any] = field(default_factory=dict)
    preprocessing_version: str = "official_v1"


@dataclass
class DecodeState:
    response_ids: torch.Tensor
    pending: torch.Tensor
    iteration: int
    window_width: int
    pressure_ema: float
    previous_candidate_ids: torch.Tensor
    previous_observed: torch.Tensor
    finish_reason: str | None = None

    def clone(self):
        return DecodeState(**{f.name: getattr(self, f.name).clone()
                             if isinstance(getattr(self, f.name), torch.Tensor)
                             else getattr(self, f.name) for f in fields(self)})


@dataclass(frozen=True)
class ForwardResult:
    response_logits: torch.Tensor
    cost: ComputeCost


@dataclass(frozen=True)
class PairedForwardResult:
    visual: ForwardResult
    ablated: ForwardResult
    execution_cost: ComputeCost


@dataclass(frozen=True)
class ProposalStats:
    base_ids: torch.Tensor
    contrast_ids: torch.Tensor
    logp_b: torch.Tensor
    logp_c: torch.Tensor
    gain_b: torch.Tensor
    gain_c: torch.Tensor
    gain_used: str
    pool_mass: torch.Tensor
    pool_size: torch.Tensor
    u_b: torch.Tensor
    u_c: torch.Tensor
    q_b: torch.Tensor
    q_c: torch.Tensor
    r_b: torch.Tensor
    r_c: torch.Tensor
    h_margin: torch.Tensor


@dataclass(frozen=True)
class GateResult:
    target: str
    pass_u: torch.Tensor
    pass_r: torch.Tensor
    eligible: torch.Tensor


@dataclass(frozen=True)
class CommitDecision:
    positions: torch.Tensor
    written_ids: torch.Tensor
    base_ids: torch.Tensor
    contrast_ids: torch.Tensor
    reason: str
    gate_target: str
    eligible_count: int


@dataclass
class RawGeneration:
    response_ids: torch.Tensor
    finish_reason: str
    decode_iterations: int
    cost: ComputeCost
    metrics: dict = field(default_factory=dict)
    first_flip: dict | None = None
    traces: list = field(default_factory=list)
    commit_log: list = field(default_factory=list)
