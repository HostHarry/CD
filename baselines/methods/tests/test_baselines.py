import copy
import gc
import json
from pathlib import Path
import tempfile
import types
import unittest
import weakref
from unittest.mock import patch

import torch

from baseline_sources import upstream_functions
from common.decoding import BaselineConfig, validate_generation
from common.native import generate_native
from common.eos import CompletedAnswer, StopAtClosedPrefix, closed_prefix_end
from common.runtime import Runtime
from vcd.decode import generate_vcd
from vigor.decode import generate_vigor
from migrate_results import validate_contract
from models.configuration_llada import LLaDAConfig
from models.modeling_llada import LLaDAModelLM
from models.modeling_mmada import MMadaModelLM


GENERATION = dict(max_new_tokens=8, steps=4, block_length=4, temperature=0.,
                  cfg_scale=0., remasking="low_confidence")


def tiny_model():
    torch.manual_seed(17)
    config = LLaDAConfig(
        vocab_size=48, embedding_size=48, d_model=32, n_heads=4,
        n_layers=6, mlp_hidden_size=64, block_type="llama", activation_type="silu",
        max_sequence_length=64, rope=True, alibi=False, weight_tying=False,
        include_bias=False, init_device="cpu", mask_token_id=47, eos_token_id=46, use_cache=False,
    )
    model = LLaDAModelLM(config).eval()
    model.mmu_generate = types.MethodType(MMadaModelLM.mmu_generate, model)
    return model


class RecordingModel:
    def __init__(self, model):
        self.model, self.config, self.inputs = model, model.config, []

    def __call__(self, tokens, **kwargs):
        self.inputs.append(tokens.clone())
        return self.model(tokens, use_cache=False)


class BaselineTests(unittest.TestCase):
    def test_text_only_prompt_has_no_image_and_methods_use_native(self):
        runtime = Runtime.__new__(Runtime)
        runtime.device = torch.device("cpu")
        runtime.early_eos = False
        runtime.model = tiny_model()
        runtime.tokenizer = unittest.mock.MagicMock()
        runtime.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        runtime.tokenizer.__len__.return_value = 20
        runtime.vq = unittest.mock.Mock()
        runtime.decode = lambda ids: {"generated_ids": ids.tolist()}
        sample = {"input_mode": "text_only", "image": None, "image_sha256": None,
                  "question": "Is this true?"}
        tokens, visual, provenance = runtime.prepare(sample)
        self.assertEqual(tokens.tolist(), [[1, 2, 3]])
        self.assertEqual(visual, [])
        self.assertIsNone(provenance["image_sha256"])
        runtime.vq.get_code.assert_not_called()
        with patch("common.runtime.generate_vcd", side_effect=AssertionError("unexpected VCD")), \
                patch("common.runtime.generate_vigor", side_effect=AssertionError("unexpected VIGOR")):
            results = [runtime.generate(sample, config, GENERATION) for config in
                       (None, {"mode": "vcd"}, {"mode": "vigor"})]
        for result, method in zip(results, ("native", "vcd", "vigor")):
            self.assertEqual(result["generated_ids"], results[0]["generated_ids"])
            self.assertEqual(result["trace"], results[0]["trace"])
            self.assertEqual(result["requested_method"], method)
            self.assertEqual(result["effective_method"], "native")
        with self.assertRaises(ValueError):
            runtime.prepare({**sample, "image": "fabricated.png"})
        with self.assertRaises(ValueError):
            validate_generation(tokens, [], GENERATION, 47)

    def test_native_text_only_matches_unmodified_generator(self):
        model = tiny_model()
        prompt = torch.tensor([[1, 2, 3, 4]])
        expected = model.mmu_generate(prompt, **GENERATION, mask_id=47)
        trace = []
        actual = generate_native(model, prompt, [], GENERATION, trace)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(len(trace), GENERATION["steps"])

    def test_hallusion_exact_matching_and_group_metrics(self):
        from common.hallusion import official_functions, rate

        extract, _ = official_functions()
        for text, expected in (
            ("YES.", "Yes"), ("No!", "No"), ("Yes, but no.", "Unknown"),
            ("", "Unknown"), ("There is a circle.", "Unknown"),
            ("Yesterday was sunny.", "Unknown"),
        ):
            self.assertEqual(extract(text), expected)
        # Two figures, two matched questions each. One unknown is a wrong answer:
        # aAcc=3/4; only one whole figure and one whole question group are correct.
        rows = [
            {"index": f"VD_illusion_1_0_{figure}_{question}", "category": "VD",
             "l2-category": "VD_illusion", "answer": "Yes", "prediction": prediction}
            for figure, question, prediction in
            [(0, 0, "Yes"), (0, 1, "Yes."), (1, 0, "YES!"), (1, 1, "Unclear")]
        ]
        scored, metrics = rate(rows)
        self.assertEqual([row["score"] for row in scored], [True, True, True, False])
        self.assertEqual(metrics["accuracy_percent"], 75.0)
        self.assertEqual(metrics["figure_accuracy_percent"], 50.0)
        self.assertEqual(metrics["question_accuracy_percent"], 50.0)
        self.assertEqual(metrics["unknown_answers"], 1)

    def test_eos_releases_intermediate_tensors_without_cyclic_gc(self):
        runtime = Runtime.__new__(Runtime)
        runtime.device = torch.device("cpu")
        runtime.early_eos = True
        runtime.model = types.SimpleNamespace(config=types.SimpleNamespace(mask_token_id=47))
        runtime.tokenizer = types.SimpleNamespace(eos_token_id=46)
        runtime.prepare = lambda sample: (torch.tensor([[1, 2, 3, 4]]), [1, 2], {})
        runtime.decode = lambda ids: {"generated_ids": ids.tolist()}
        references = []

        def generator(model, tokens, visual, generation, trace):
            intermediate = torch.ones(100)
            references.append(weakref.ref(intermediate))
            return model(torch.tensor([[1, 2, 3, 4, 5, 46, 47, 47, 47, 47, 47, 47]]))

        enabled = gc.isenabled()
        gc.disable()
        try:
            with patch("common.runtime.generate_native", generator):
                for _ in range(4):
                    result = runtime.generate({}, None, GENERATION)
                    self.assertTrue(result["early_stopped"])
                    self.assertEqual(result["generated_ids"], [5, 46])
                    self.assertIsNone(references[-1](), "exception traceback retained a forward tensor")
        finally:
            if enabled:
                gc.enable()
            gc.collect()

    def test_eos_requires_committed_prefix_and_ignores_prompt_eos(self):
        self.assertIsNone(closed_prefix_end([47, 46, 47], 46, 47))
        self.assertIsNone(closed_prefix_end([5, 47, 47], 46, 47))
        self.assertEqual(closed_prefix_end([5, 46, 47], 46, 47), 2)
        self.assertEqual(closed_prefix_end([46, 47], 46, 47), 1)
        model = unittest.mock.Mock()
        model.config.mask_token_id = 47
        logits = torch.zeros(1, 1, 48)
        logits[..., 46] = 100.
        model.return_value = types.SimpleNamespace(logits=logits)
        proxy = StopAtClosedPrefix(model, 2, 46)
        # Even a model predicting EOS cannot stop while actual state is masked.
        proxy(torch.tensor([[46, 47, 47, 47]]))
        proxy(torch.tensor([[46, 47, 47, 46]]))
        self.assertEqual(model.call_count, 2)
        state = torch.tensor([[46, 47, 5, 46, 47]])
        with self.assertRaises(CompletedAnswer) as caught:
            proxy(state)
        self.assertEqual(model.call_count, 2)
        self.assertEqual(caught.exception.answer_end, 2)
        self.assertTrue(torch.equal(caught.exception.state, state))

    def test_safe_eos_matches_all_generators_and_no_eos_falls_back(self):
        class FixedLogits:
            device = torch.device("cpu")
            config = types.SimpleNamespace(mask_token_id=47)
            mmu_generate = MMadaModelLM.mmu_generate

            def __init__(self, eos):
                self.eos = eos

            def __call__(self, tokens, **kwargs):
                logits = torch.zeros(*tokens.shape, 48)
                for pos in range(4, tokens.shape[1]):
                    token = 5 if pos == 4 or not self.eos else 46
                    logits[:, pos, token] = 2 if pos == 4 else 12 - pos
                return types.SimpleNamespace(logits=logits)

        prompt = torch.tensor([[1, 2, 3, 4]])
        distorted = torch.tensor([[1, 8, 9, 4]])
        functions = [
            lambda model, trace: generate_native(model, prompt, [1, 2], GENERATION, trace),
            lambda model, trace: generate_vcd(model, prompt, distorted, [1, 2], GENERATION,
                                               BaselineConfig("vcd"), trace),
            lambda model, trace: generate_vigor(model, prompt, [1, 2], GENERATION, trace),
        ]
        for eos in (True, False):
            for run in functions:
                model = FixedLogits(eos)
                full_trace, early_trace = [], []
                full = run(model, full_trace)
                proxy = StopAtClosedPrefix(model, 4, 46)
                if eos:
                    with self.assertRaises(CompletedAnswer) as caught:
                        run(proxy, early_trace)
                    self.assertLess(len(early_trace), 4)
                    end = 4 + caught.exception.answer_end
                    self.assertTrue(torch.equal(caught.exception.state[:, :end], full[:, :end]))
                    # Every commit and score up to the stopping boundary is identical.
                    self.assertEqual(early_trace, full_trace[:len(early_trace)])
                else:
                    early = run(proxy, early_trace)
                    self.assertTrue(torch.equal(early, full))
                    self.assertEqual(early_trace, full_trace)

    def test_vcd_original_noise_and_apc(self):
        functions = upstream_functions()
        pixels = torch.ones(3, 8, 8)
        torch.manual_seed(42)
        first = functions["noise"](pixels, 500)
        torch.manual_seed(42)
        self.assertTrue(torch.equal(first, functions["noise"](pixels, 500)))
        self.assertFalse(torch.equal(first, pixels))
        # Token 2 would win the contrast but must be removed by original-image APC.
        normal = torch.tensor([[4., 3., -10.]])
        distorted = torch.tensor([[4., -1., -100.]])
        result = functions["vcd"](normal, distorted, 1., .1)
        self.assertEqual(result[0, :2].tolist(), [4., 7.])
        self.assertTrue(torch.isneginf(result[0, 2]))
        self.assertEqual(result.argmax(-1).item(), 1)
        # Exactly-on-cutoff tokens are retained (strict '<' in author code).
        result = functions["vcd"](torch.zeros(1, 4), torch.zeros(1, 4), 1., 1.)
        self.assertTrue(torch.isfinite(result).all())
        with self.assertRaises(ValueError):
            BaselineConfig(mode="vcd", cd_beta=0)

    def test_vigor_observation_preserves_complete_author_loop(self):
        prompt = torch.tensor([[1, 2, 3, 4]])
        for ties in (False, True):
            model = tiny_model()
            if ties:
                with torch.no_grad():
                    model.model.transformer.ff_out.weight.zero_()
            ours, original = RecordingModel(model), RecordingModel(model)
            trace = []
            result = generate_vigor(ours, prompt, [1, 2], GENERATION, trace)
            x = torch.full((1, 12), 47, dtype=torch.long)
            x[:, :4] = prompt
            expected = upstream_functions()["vigor"](
                original, x, steps=4, gen_length=8, block_length=4, temperature=0.,
                mask_id=47, code_start=4, probe_interval=1, img_start=1, img_end=3)
            self.assertTrue(torch.equal(result, expected))
            self.assertEqual(len(ours.inputs), 8)
            for left, right in zip(ours.inputs, original.inputs):
                self.assertTrue(torch.equal(left, right))
            self.assertEqual(len(trace), 4)
            for index, step in enumerate(trace):
                before = ours.inputs[2 * index].clone()
                before[0, step["committed_positions"][0]] = torch.tensor(step["committed_tokens"][0])
                after = ours.inputs[2 * index + 2] if index < 3 else result
                self.assertTrue(torch.equal(before, after))
                self.assertEqual(len(step["committed_positions"][0]), 2)
                self.assertEqual(step["remaining_masks"], int((after[:, 4:] == 47).sum()))
                self.assertTrue((ours.inputs[2 * index + 1][0, 1:3] == 47).all())
                self.assertEqual(ours.inputs[2 * index + 1][0, 0], 1)
                self.assertEqual(ours.inputs[2 * index + 1][0, 3], 4)

    def test_vigor_orders_by_gain_without_contrasting_candidates(self):
        class EvidenceModel:
            config = types.SimpleNamespace(mask_token_id=7)

            def __call__(self, tokens, **kwargs):
                probability = torch.full((*tokens.shape, 8), 1e-6)
                no_image = tokens[0, 1] == 7
                # Both positions predict token 0 in the normal branch. Position 3
                # has greater confidence but position 4 has greater visual support.
                for pos, img, ablated in ((3, .9, .89), (4, .8, .1)):
                    p = ablated if no_image else img
                    probability[0, pos, 0] = p
                    probability[0, pos, 1] = 1 - p
                return types.SimpleNamespace(logits=probability.log())
        trace = []
        generation = {**GENERATION, "max_new_tokens": 2, "steps": 2, "block_length": 2}
        result = generate_vigor(EvidenceModel(), torch.tensor([[1, 2, 3]]), [1], generation, trace)
        self.assertEqual(trace[0]["committed_positions"], [[4]])
        self.assertEqual(trace[0]["committed_tokens"], [[0]])
        self.assertEqual(result[0, 3:].tolist(), [0, 0])

    def test_vcd_shares_current_text_and_keeps_schedule(self):
        model = RecordingModel(tiny_model())
        prompt, distorted = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[1, 8, 9, 4]])
        trace = []
        output = generate_vcd(model, prompt, distorted, [1, 2], GENERATION, BaselineConfig("vcd"), trace)
        self.assertEqual(len(trace), 4)
        self.assertEqual(len(model.inputs), 8)
        for index, step in enumerate(trace):
            normal, reference = model.inputs[2 * index:2 * index + 2]
            self.assertTrue(torch.equal(reference[:, 1:3], distorted[:, 1:3]))
            self.assertTrue(torch.equal(normal[:, 4:], reference[:, 4:]))
            self.assertEqual(len(step["committed_positions"][0]), 2)
            self.assertLess(max(step["committed_positions"][0]), step["block_end"])
            after = model.inputs[2 * index + 2] if index < 3 else output
            normal = normal.clone()
            normal[0, step["committed_positions"][0]] = torch.tensor(step["committed_tokens"][0])
            self.assertTrue(torch.equal(normal, after))
        with self.assertRaises(ValueError):
            validate_generation(prompt, [1, 2], {**GENERATION, "temperature": 1.}, 47)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            bad = types.SimpleNamespace(config=types.SimpleNamespace(mask_token_id=47))
            class Nonfinite:
                config = bad.config
                def __call__(self, tokens, **kwargs):
                    return types.SimpleNamespace(logits=torch.full((*tokens.shape, 48), float("nan")))
            generate_vigor(Nonfinite(), prompt, [1, 2], GENERATION, [])

    def test_native_reuse_rejects_protocol_changes(self):
        reference = {
            **{name: "same" for name in (
                "upstream_revision", "environment", "generation", "samples_sha256",
                "expected_rows", "selected_rows", "limit", "benchmark", "excluded_samples_sha256", "checkpoints", "seed_protocol")},
            "config": {name: "same" for name in ("checkpoint", "vq_checkpoint", "resolution", "resize", "seed")},
            "sources": {name: "same" for name in ("models/modeling_llada.py", "models/configuration_llada.py",
                        "models/modeling_magvitv2.py", "training/prompting_utils.py", "training/utils.py")},
            "methods": {"native": None},
        }
        target = copy.deepcopy(reference)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "data/relocation.json").write_text(json.dumps({
                "samples.jsonl": {"old_sha256": "same", "new_sha256": "same"},
                "calibration.jsonl": {"old_sha256": "same", "new_sha256": "same"},
            }))
            with patch("migrate_results.ROOT", root), patch("migrate_results.sha256", return_value="same"):
                validate_contract(reference, target, "native", "samples.jsonl")
                for field in ("generation", "samples_sha256", "checkpoints", "seed_protocol"):
                    changed = copy.deepcopy(target)
                    changed[field] = "different"
                    with self.assertRaises(ValueError):
                        validate_contract(reference, changed, "native", "samples.jsonl")
                changed = copy.deepcopy(target)
                changed["config"]["resize"] = "squash"
                with self.assertRaises(ValueError):
                    validate_contract(reference, changed, "native", "samples.jsonl")

    def test_native_observer_keeps_official_output_and_schedule(self):
        model = tiny_model()
        prompt = torch.tensor([[1, 2, 3, 4]])
        for ties in (False, True):
            if ties:
                with torch.no_grad():
                    model.model.transformer.ff_out.weight.zero_()
            expected = model.mmu_generate(prompt, **GENERATION, mask_id=47)
            trace = []
            actual = generate_native(model, prompt, [1, 2], GENERATION, trace)
            self.assertTrue(torch.equal(expected, actual))
            state = torch.full((1, 12), 47, dtype=torch.long)
            state[:, :4] = prompt
            for step in trace:
                self.assertEqual(len(step["committed_positions"][0]), 2)
                state[0, step["committed_positions"][0]] = torch.tensor(step["committed_tokens"][0])
            self.assertTrue(torch.equal(state, actual))


if __name__ == "__main__":
    unittest.main()
