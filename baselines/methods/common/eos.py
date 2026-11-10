"""Stop only after an immutable generated prefix, including EOS, is committed."""


class CompletedAnswer(Exception):
    def __init__(self, state, answer_end):
        super().__init__("all generated positions through EOS are committed")
        self.state = state
        self.answer_end = answer_end


def closed_prefix_end(ids, eos_id, mask_id):
    """Exclusive end including EOS, or None if the prefix is not yet complete."""
    try:
        end = ids.index(eos_id) + 1
    except ValueError:
        return None
    return end if mask_id not in ids[:end] else None


class StopAtClosedPrefix:
    """Check actual state before a forward, never predicted candidate logits.

    All supported generators only overwrite MASK positions. Therefore a
    MASK-free prefix ending in EOS cannot change in the remaining schedule.
    This proxy leaves their selection rules and original functions untouched.
    """
    def __init__(self, model, prompt_len, eos_id):
        self.model, self.prompt_len, self.eos_id = model, prompt_len, eos_id

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, tokens, *args, **kwargs):
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError("safe EOS stopping requires batch=1")
        end = closed_prefix_end(tokens[0, self.prompt_len:].tolist(),
                                self.eos_id, self.model.config.mask_token_id)
        if end is not None:
            raise CompletedAnswer(tokens.clone(), end)
        return self.model(tokens, *args, **kwargs)


def answer_ids(result):
    """Canonical scored token sequence, including EOS when present."""
    end = result["answer_token_length"] + int(result["eos_found"])
    return result["generated_ids"][:end]
