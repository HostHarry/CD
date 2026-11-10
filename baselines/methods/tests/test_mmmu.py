import json
import random
import unittest

from common.mmmu_protocol import DATA, INSTRUCTION, image_key, user_content
from common.mmmu_score import OfficialScorer


class MMMUProtocolTests(unittest.TestCase):
    def test_reference_selection_is_sorted_and_includes_options(self):
        options = {"A": "<image 1> red", "B": "<image3> blue"}
        question = "<image 2> Which <image> is <image2> correct?"
        self.assertEqual(image_key(question, options), "image_1")
        self.assertEqual(
            user_content(question, options),
            INSTRUCTION + "\n Which  is  correct?\nA.  red\nB.  blue\n")
        self.assertEqual(image_key("See <image2>.", {}), "image_1")
        self.assertEqual(user_content("What?", {}), INSTRUCTION + "\nWhat?\n")

    def test_two_shards_cover_reference_validation_without_gold(self):
        rows = list(map(json.loads, (DATA / "samples.jsonl").read_text().splitlines()))
        shards = [list(map(json.loads, (DATA / f"shard_{i}.jsonl").read_text().splitlines()))
                  for i in range(2)]
        self.assertEqual([len(s) for s in shards], [450, 450])
        self.assertEqual(shards, [rows[::2], rows[1::2]])
        self.assertEqual(len({s["id"] for shard in shards for s in shard}), 900)
        self.assertEqual(sum(s["source_image_count"] > 1 for s in rows), 43)
        self.assertEqual(sum(s["question_type"] == "open" for s in rows), 53)
        self.assertTrue(all(s["selected_image"] == "image_1" for s in rows))
        self.assertTrue(all(not {"answer", "label", "explanation", "target"} & s.keys() for s in rows))

    def test_official_random_fallback_is_explicit_and_reproducible(self):
        state = random.getstate()
        scorer = OfficialScorer()
        sample = {"id": "validation_fixture_1", "question_type": "multiple-choice",
                  "options": {"A": "red", "B": "blue", "C": "green", "D": "yellow"}}
        first = scorer.score(sample, "B", "I cannot tell.")
        self.assertTrue(first["used_random_guess"])
        self.assertIsNone(first["no_guess_prediction"])
        self.assertFalse(first["no_guess_correct"])
        self.assertEqual(first, scorer.score(sample, "B", "I cannot tell."))
        self.assertEqual(random.getstate(), state)
        explicit = scorer.score(sample, "B", "The answer is (B).")
        self.assertTrue(explicit["official_correct"])
        self.assertFalse(explicit["used_random_guess"])

    def test_official_rules_keep_think_text_and_support_open_aliases(self):
        scorer = OfficialScorer()
        sample = {"id": "validation_fixture_2", "question_type": "multiple-choice",
                  "options": {"A": "red", "B": "blue"}}
        # The official parser prioritizes bracketed letters, even in reasoning.
        result = scorer.score(sample, "A", "<think>Considering (B).</think>\nA")
        self.assertEqual(result["official_prediction"], "B")
        self.assertFalse(result["official_correct"])
        sample["question_type"] = "open"
        result = scorer.score(sample, "['$MgS$', 'MgS']", "The answer is MgS.")
        self.assertTrue(result["gold_aliases_decoded"])
        self.assertTrue(result["official_correct"])
        self.assertTrue(result["no_guess_correct"])
        self.assertTrue(scorer.score(sample, "['24/7', '3.429']", "The answer is 3.429.")["official_correct"])


if __name__ == "__main__":
    unittest.main()
