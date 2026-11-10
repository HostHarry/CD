def normalize_output(response_ids, tokenizer, eos_ids, normalizer_id="first_eos_v1"):
    if normalizer_id != "first_eos_v1":
        raise ValueError("Unknown normalizer")
    ids = list(response_ids)
    first = next((i for i, token in enumerate(ids) if token in eos_ids), None)
    visible = ids if first is None else ids[:first]
    text = tokenizer.decode(visible, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    return dict(text=text, first_eos_position=first, emitted_tokens_before_eos=len(visible),
                full_buffer_length=len(ids), normalizer_id=normalizer_id)
