from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest

from multigpu import connect, prepare_tasks, claim_task, finish_task, validate_fingerprints, token_fingerprint


class MultiGPUTests(unittest.TestCase):
    def test_early_eos_fingerprint_preserves_answer_and_prompt(self):
        full = {"input": {"prompt_ids": [1, 2]}, "generated_ids": [5, 46, 46, 46],
                "answer_token_length": 1, "eos_found": True}
        stopped = {**full, "generated_ids": [5, 46]}
        self.assertNotEqual(token_fingerprint(full), token_fingerprint(stopped))
        self.assertEqual(token_fingerprint(full, True), token_fingerprint(stopped, True))
        self.assertNotEqual(token_fingerprint(full, True),
                            token_fingerprint({**stopped, "generated_ids": [6, 46]}, True))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "predictions.sqlite"
        self.connection = connect(self.path)
        self.connection.execute(
            "CREATE TABLE predictions (method TEXT,input_key TEXT,payload TEXT,PRIMARY KEY(method,input_key))")
        self.tasks = [(method, str(i)) for i in range(12) for method in ("native", "vcd")]
        with self.connection:
            self.connection.execute("INSERT INTO predictions VALUES ('native','0',?)", ('{"preserved":true}',))

    def tearDown(self):
        self.connection.close()
        self.directory.cleanup()

    def test_parallel_claims_preserve_existing_results(self):
        self.assertEqual(prepare_tasks(self.connection, self.tasks), 1)

        def consume(worker):
            connection = connect(self.path)
            claimed = []
            try:
                while (task := claim_task(connection, worker)) is not None:
                    finish_task(connection, worker, task, {"worker": worker, "key": task[1]})
                    claimed.append(task)
            finally:
                connection.close()
            return claimed

        with ThreadPoolExecutor(2) as pool:
            groups = list(pool.map(consume, ("0", "1")))
        claimed = [task for group in groups for task in group]
        self.assertEqual(len(claimed), len(set(claimed)))
        self.assertEqual(set(claimed), set(self.tasks) - {("native", "0")})
        self.assertEqual(self.connection.execute(
            "SELECT payload FROM predictions WHERE method='native' AND input_key='0'").fetchone()[0],
            '{"preserved":true}')
        self.assertEqual(self.connection.execute("SELECT count(*) FROM predictions").fetchone()[0], 24)
        self.assertEqual(self.connection.execute("SELECT count(*) FROM gpu_tasks WHERE state!='done'").fetchone()[0], 0)

    def test_interrupted_claim_is_retried_without_overwriting(self):
        prepare_tasks(self.connection, self.tasks)
        task = claim_task(self.connection, "old")
        prepare_tasks(self.connection, self.tasks)
        self.assertEqual(claim_task(self.connection, "new"), task)
        with self.assertRaises(ValueError):
            finish_task(self.connection, "old", task, {"wrong": True})
        finish_task(self.connection, "new", task, {"correct": True})
        with self.assertRaises(ValueError):
            finish_task(self.connection, "new", task, {"duplicate": True})
        value = self.connection.execute(
            "SELECT payload FROM predictions WHERE method=? AND input_key=?", task).fetchone()[0]
        self.assertEqual(json.loads(value), {"correct": True})

    def test_reject_unknown_predictions_and_duplicate_tasks(self):
        with self.assertRaises(ValueError):
            prepare_tasks(self.connection, [("vcd", "0")])
        with self.assertRaises(ValueError):
            prepare_tasks(self.connection, self.tasks + [self.tasks[0]])

    def test_cross_gpu_and_preserved_prediction_checks(self):
        ready = [{"fingerprints": {"native": "a", "vcd": "b"}}] * 2
        validate_fingerprints(ready, {"native": "a"})
        with self.assertRaises(ValueError):
            validate_fingerprints(ready, {"native": "different"})
        with self.assertRaises(ValueError):
            validate_fingerprints([ready[0], {"fingerprints": {"native": "a", "vcd": "other"}}], {})


if __name__ == "__main__":
    unittest.main()
