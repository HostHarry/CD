"""Fetch, verify and load the pinned authors' inference code without its CLI imports."""
import argparse
import ast
from functools import cache
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
PINS = [ROOT / name / "source.json" for name in ("vcd", "vigor")]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(fetch=False):
    sources = {str(pin.relative_to(ROOT)): digest(pin) for pin in PINS}
    for pin in PINS:
        entry = json.loads(pin.read_text())
        for relative, expected in entry["files"].items():
            path = pin.parent / "upstream" / relative
            if fetch and not path.exists():
                repo = entry["repository"].removeprefix("https://github.com/")
                url = f"https://raw.githubusercontent.com/{repo}/{entry['revision']}/{relative}"
                with urllib.request.urlopen(url, timeout=60) as response:
                    content = response.read()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError(f"download hash mismatch: {url}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            if not path.exists() or digest(path) != expected:
                raise ValueError(f"missing/changed upstream source: {path}; run baseline_sources.py --fetch")
            sources[str(path.relative_to(ROOT))] = expected
    return sources


def compile_function(node, filename, namespace):
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(filename), "exec"), namespace)
    return namespace[node.name]


@cache
def upstream_functions():
    # Only function definitions / the three VCD logit assignments are loaded.
    # No upstream CLI, import path changes, model downloads or HF monkeypatches run.
    verify_sources()
    import numpy as np
    import torch
    from models.modeling_mmada import add_gumbel_noise, get_num_transfer_tokens

    paths = {
        "noise": ROOT / "vcd/upstream/vcd_utils/vcd_add_noise.py",
        "vcd": ROOT / "vcd/upstream/vcd_utils/vcd_sample.py",
        "vigor": ROOT / "vigor/upstream/scripts/run_amber_lumina.py",
    }
    namespace = dict(torch=torch, np=np, F=torch.nn.functional,
                     add_gumbel_noise=add_gumbel_noise, get_num_transfer_tokens=get_num_transfer_tokens)
    functions = {}
    for key, name in (("noise", "add_diffusion_noise"),
                      ("vigor", "generate_text_understanding_visual_ablation")):
        tree = ast.parse(paths[key].read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        functions[key] = compile_function(node, paths[key], dict(namespace))
    tree = ast.parse(paths["vcd"].read_text())
    selected = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in {"cutoff", "diffs"} or (
            target.id == "cd_logits" and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "masked_fill"
        ):
            if target.id in selected:
                raise ValueError("ambiguous VCD kernel extraction")
            selected[target.id] = node
    node = ast.parse(
        "def vcd_logits(next_token_logits, next_token_logits_cd, cd_alpha, cd_beta):\n"
        "    pass\n").body[0]
    node.body = [selected[name] for name in ("cutoff", "diffs", "cd_logits")]
    node.body.append(ast.Return(value=ast.Name(id="cd_logits", ctx=ast.Load())))
    functions["vcd"] = compile_function(node, paths["vcd"], dict(namespace))
    return functions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    print(json.dumps(verify_sources(parser.parse_args().fetch), indent=2))
