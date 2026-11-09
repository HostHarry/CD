import torch
from cdapc_lab.models.base import visual_ablation_bias


def test_noncontiguous_visual_block_reaches_real_sdpa():
    torch.manual_seed(7)
    visual = torch.tensor([True, False, True, False])
    q, k, v = (torch.randn(1, 2, 6, 4) for _ in range(3))
    changed = v.clone()
    changed[:, :, [0, 2]] += 100
    run = lambda value, mask: torch.nn.functional.scaled_dot_product_attention(q, k, value, attn_mask=mask)
    bias = visual_ablation_bias(visual, 2)
    texts = [1, 3, 4, 5]
    assert not torch.allclose(run(v, None)[:, :, texts], run(changed, None)[:, :, texts])
    torch.testing.assert_close(run(v, bias)[:, :, texts], run(changed, bias)[:, :, texts], rtol=0, atol=0)
    assert bias[0, 0, 0, 2] == 0 and bias[0, 0, 1, 2] == -torch.inf
