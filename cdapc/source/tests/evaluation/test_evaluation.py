from cdapc_lab.evaluation.parsers import strict_parse
from cdapc_lab.evaluation.normalize import normalize_output
from cdapc_lab.evaluation.scorers import load_official_scorer, score_text
from pathlib import Path
import inspect
import pytest


@pytest.mark.parametrize("text,answer", [
    ("Analysis of option B. The final result is (A).", "A"),
    ("The answer is a red ball. Final: (B).", "B"),
    (r"Earlier \boxed{A}; Final answer: (C)", "C"),
    ("We discuss A/B/C.\n(D)", "D"), ("B", "B"),
    (r"\boxed{B}！", "B"), ("Final answer: (B)。", "B")])
def test_final_answer(text, answer):
    parsed = strict_parse(text, "ABCD")
    assert (parsed.prediction, parsed.status) == (answer, "parsed")


@pytest.mark.parametrize("text,status", [
    ("Final answer: (A) or (B)", "ambiguous"), ("(A) or (B)", "ambiguous"),
    (r"Final answer: \boxed{A", "invalid"), ("Final: (Z)", "invalid"),
    ("option B is plausible", "unparsed"), ("a red ball", "unparsed"),
    ("", "unparsed"), ("Final answer: (B)!!", "unparsed")])
def test_no_guess(text, status):
    assert strict_parse(text, "ABCD").status == status


class Tokenizer:
    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


@pytest.mark.parametrize("ids,text,first", [([1, 2, 9, 3, 9], "1 2", 2), ([9, 1], "", 0), ([1, 2], "1 2", None)])
def test_eos(ids, text, first):
    result = normalize_output(ids, Tokenizer(), (9,))
    assert result["text"] == text and result["first_eos_position"] == first
    assert result["full_buffer_length"] == len(ids)


def test_gold_isolation():
    from cdapc_lab.data.records import build_question
    assert "gold" not in inspect.signature(strict_parse).parameters
    assert list(inspect.signature(build_question).parameters) == ["sample"]


def test_official_receives_original_text():
    seen = []
    def scorer(text, choices, gold):
        seen.append(text)
        return False
    out = score_text("Reasoning. Final: (B)", {"A": "a", "B": "b"}, "B", scorer)
    assert seen == ["Reasoning. Final: (B)"]
    assert out["strict_accuracy"] == 1 and out["official"] == 0


def test_pinned_official_function():
    source = Path(__file__).parents[2] / "upstream/m3cot/utils/metric.py"
    if not source.is_file():
        pytest.skip("Official source checkout absent")
    scorer = load_official_scorer(source)
    assert scorer("Final answer: (B)", ["cat", "dog"], "B")
    assert not scorer("no final", ["cat", "dog"], "B")
