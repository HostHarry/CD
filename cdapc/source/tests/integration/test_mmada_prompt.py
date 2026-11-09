import json
from pathlib import Path
import pytest

from cdapc_lab.models.mmada import OFFICIAL_THINKING_INSTRUCTION


def test_mmada_thinking_instruction_matches_official():
    root = Path(__file__).parents[2] / "upstream/mmada"
    prompts_file = root / "mmu_validation/prompts_with_vqa.json"
    app_file = root / "app.py"
    if not prompts_file.is_file() or not app_file.is_file():
        pytest.skip("Official source checkout absent")
    prompts = json.loads(prompts_file.read_text())
    content = prompts[0]["messages"][0]["content"]
    assert content.split("\n", 1)[0] == OFFICIAL_THINKING_INSTRUCTION
    assert OFFICIAL_THINKING_INSTRUCTION in app_file.read_text()
