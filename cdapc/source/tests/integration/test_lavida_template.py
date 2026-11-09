from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace


def test_official_llada_template_needs_no_unrelated_download(monkeypatch):
    source = Path(__file__).parents[2] / "upstream/lavida/llava/conversation.py"
    def forbidden(*args, **kwargs):
        raise AssertionError("No tokenizer downloads during conversation import")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=forbidden)))
    spec = importlib.util.spec_from_file_location("verified_conversation", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    conversation = module.conv_templates["llada"].copy()
    received = []
    conversation.tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: received.append((messages, kwargs)) or "prompt")
    conversation.append_message(conversation.roles[0], "<image>\nquestion")
    conversation.append_message(conversation.roles[1], None)
    assert conversation.get_prompt() == "prompt"
    assert received[0][0][-1]["content"] == "<image>\nquestion"
    assert received[0][1] == {"tokenize": False, "add_generation_prompt": True}
