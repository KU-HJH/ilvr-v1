import pytest
import torch

from transformers.models.qwen2_5_vl.stage2_latent_rollout import (
    LatentStructureError,
    LatentTokenIds,
    parse_latent_spans,
)

TOKENS = LatentTokenIds(pad_token_id=90, start_token_id=91, end_token_id=92)


def test_parse_multiple_latent_spans_in_order():
    ids = torch.tensor([1, 91, 90, 90, 92, 2, 91, 90, 90, 92, 3])
    spans = parse_latent_spans(ids, token_ids=TOKENS, latent_size=2)
    assert [(s.start_position, s.latent_positions, s.end_position) for s in spans] == [
        (1, (2, 3), 4),
        (6, (7, 8), 9),
    ]


@pytest.mark.parametrize(
    "ids, message",
    [
        ([1, 91, 90, 90], "missing latent_end"),
        ([1, 92], "latent_end without latent_start"),
        ([91, 90, 91, 90, 92, 92], "nested latent_start"),
        ([91, 90, 92], "expected 2 latent_pad tokens"),
        ([91, 90, 7, 92], "non-latent_pad token"),
    ],
)
def test_reject_malformed_latent_structure(ids, message):
    with pytest.raises(LatentStructureError, match=message):
        parse_latent_spans(torch.tensor(ids), token_ids=TOKENS, latent_size=2)
