"""MMMU-val prompt construction aligned with WINO's MMaDA MMMU task."""
import re

_PLACEHOLDER = re.compile(r"<image \d+>|<image\d+>|<image>")


def strip_image_placeholders(text):
    """WINO models/mmada.py context cleanup: <image>, <image n>, <imagen> -> ''."""
    text = text.replace("<image>", "")
    text = re.sub(r"<image \d+>", "", text)
    return re.sub(r"<image\d+>", "", text)


def options_letters(options):
    return "\n".join(f"{letter}. {option}" for letter, option in options.items())


def mmmu_context(question, options):
    """WINO construct_prompt + wrapper placeholder removal."""
    if options:
        context = f"{question}\n{options_letters(options)}\n"
    else:
        context = f"{question}\n"
    return strip_image_placeholders(context)


def mmmu_judge_question(question, options):
    """WINO's judge-side construct_prompt: no placeholder removal, no instruction."""
    if options:
        return f"{question}\n{options_letters(options)}\n"
    return f"{question}\n"


def choose_image(context_source_text):
    """lmms-eval mmmu_doc_to_visual: first distinct <image n> placeholder (sorted); else image_1."""
    tokens = sorted(set(re.findall(r"<image \d+>", context_source_text)))
    if tokens:
        return tokens[0].strip("<>").replace(" ", "_")
    return "image_1"
