from __future__ import annotations
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ParsedAnswer:
    prediction: str | None
    status: str
    evidence_span: tuple[int, int] | None
    rule_id: str
    parser_revision: str = "strict_final_v1"


def strict_parse(text, allowed_options):
    allowed = set(allowed_options)
    if not allowed or any(re.fullmatch(r"[A-Z]", x) is None for x in allowed):
        raise ValueError("strict_final_v1 requires uppercase MCQ labels")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    final = list(re.finditer(r"(?:final\s+answer\s*:|\bfinal\s*:|the\s+final\s+result\s+is\s*)", text, re.I))
    start = final[-1].end() if final else text.rfind("\n") + 1
    region = text[start:].strip()
    atom = r"(?:\(([A-Z])\)|\\boxed\{([A-Z])\}|([A-Z]))"
    match = re.fullmatch(atom + r"[.。!！]?", region)
    rule = "explicit_final" if final else "standalone_final"
    # A trailing parenthesized/boxed answer is allowed; prose letters are not.
    if not match and not final:
        candidates = list(re.finditer(r"\(([A-Z])\)|\\boxed\{([A-Z])\}", region))
        if candidates and re.fullmatch(r"[.。!！]?", region[candidates[-1].end():].strip()):
            labels = {next(g for g in m.groups() if g) for m in candidates}
            if len(labels) > 1:
                return ParsedAnswer(None, "ambiguous", (start, len(text)), "conflicting_final_region")
            match = re.fullmatch(atom + r"[.。!！]?", region[candidates[-1].start():])
            start += text[start:].find(region) + candidates[-1].start()
            rule = "trailing_delimited"
    if match:
        label = next(g for g in match.groups() if g)
        return ParsedAnswer(label if label in allowed else None,
                            "parsed" if label in allowed else "invalid",
                            (start, len(text)), rule)
    labels = set(re.findall(r"(?<!\w)[A-Z](?!\w)", region)) if final else set()
    if len(labels) > 1:
        return ParsedAnswer(None, "ambiguous", (start, len(text)), "conflicting_final_region")
    status = "invalid" if "\\boxed{" in region or (labels - allowed) else "unparsed"
    return ParsedAnswer(None, status, None, "no_complete_final_answer")


def strict_yes_no(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    hits = [(m.group(), m.span()) for m in re.finditer(r"[a-z]+", text.lower()) if m.group() in ("yes", "no")]
    kinds = {word for word, _ in hits}
    span = hits[0][1] if hits else None
    if kinds == {"yes"}:
        return ParsedAnswer("yes", "parsed", span, "single_polarity", "strict_yes_no_v1")
    if kinds == {"no"}:
        return ParsedAnswer("no", "parsed", span, "single_polarity", "strict_yes_no_v1")
    if kinds:
        return ParsedAnswer(None, "ambiguous", span, "conflicting_polarity", "strict_yes_no_v1")
    return ParsedAnswer(None, "unparsed", None, "no_polarity_word", "strict_yes_no_v1")


def reasoned_yes_no(text):
    """Read only a completed final-answer region; never infer polarity from reasoning."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    revision = "pope_final_yes_no_v1"
    opening, closing = text.rfind("<think>"), text.rfind("</think>")
    if opening >= 0 and closing < opening:
        return ParsedAnswer(None, "unparsed", None, "unclosed_think", revision)
    start = closing + len("</think>") if closing >= 0 else 0
    markers = list(re.finditer(r"\bfinal\s+answer\s*:", text[start:], re.I))
    if markers:
        start += markers[-1].end()
    match = re.fullmatch(r"\s*(yes|no)[.!]?\s*", text[start:], re.I)
    if match:
        return ParsedAnswer(match.group(1).lower(), "parsed",
                            (start + match.start(1), start + match.end(1)),
                            "explicit_final" if markers else "final_region", revision)
    kinds = set(re.findall(r"\b(?:yes|no)\b", text[start:].lower()))
    return ParsedAnswer(None, "ambiguous" if len(kinds) > 1 else "unparsed", None,
                        "no_unique_final_answer", revision)
