import torch

from tests.stage2.helpers import tiny_stage2_qwen


def test_eager_decoder_honors_shifted_target_edge_mask():
    """Catches a decoder change that drops an explicit 4-D additive edge mask."""
    torch.manual_seed(17)
    model = tiny_stage2_qwen(attn_implementation="eager").eval()
    decoder = model.model

    fixed_embeddings = torch.tensor(
        [
            [
                [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
                [0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10],
                [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
                [0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15],
                [0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82],
                [0.82, 0.72, 0.62, 0.52, 0.42, 0.32, 0.22, 0.12],
            ]
        ],
        dtype=torch.float32,
    )
    position_ids = torch.tensor(
        [[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]]
    )
    minimum = torch.finfo(fixed_embeddings.dtype).min
    causal_mask = torch.full((1, 1, 6, 6), minimum, dtype=fixed_embeddings.dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    shifted_mask = causal_mask.clone()

    # Logical targets [4, 6) are predicted by physical decoder rows [3, 5).
    shifted_mask[:, :, 3:5, 1:3] = minimum

    position_ids_seen = []

    def record_position_ids(_module, _args, kwargs):
        position_ids_seen.append(kwargs["position_ids"].detach().clone())

    handle = decoder.register_forward_pre_hook(record_position_ids, with_kwargs=True)
    try:
        with torch.no_grad():
            baseline = decoder(
                inputs_embeds=fixed_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
            shifted = decoder(
                inputs_embeds=fixed_embeddings,
                attention_mask=shifted_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
    finally:
        handle.remove()

    baseline_distribution = model.lm_head(baseline.last_hidden_state).softmax(dim=-1)
    shifted_distribution = model.lm_head(shifted.last_hidden_state).softmax(dim=-1)

    assert len(position_ids_seen) == 2
    torch.testing.assert_close(position_ids_seen[0], position_ids)
    torch.testing.assert_close(position_ids_seen[1], position_ids)
    torch.testing.assert_close(position_ids_seen[0], position_ids_seen[1])
    assert not torch.allclose(
        baseline_distribution[:, 3:5], shifted_distribution[:, 3:5], atol=1e-7, rtol=1e-6
    )
    torch.testing.assert_close(
        baseline_distribution[:, :3], shifted_distribution[:, :3], atol=1e-7, rtol=1e-6
    )
