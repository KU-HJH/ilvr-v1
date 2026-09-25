import pytest
from evaluation.latent_image_layout import (
    LayoutTokenIds, build_layout, query_regions, latent_query_positions,
    SINK, TEMPLATE, IMAGE, QUESTION, REASONING, LATENT, PREV_BLOCK, CURRENT_BLOCK,
)

T = LayoutTokenIds(image=80, vision_start=81, vision_end=83, im_end=84,
                   latent_start=91, latent_pad=90, latent_end=92, answer_marker=(70, 71))
# prompt: [sys 1, 2] vs img img ve [q 5, 6] im_end [tmpl 7, 8]  -> length 11
PROMPT = [1, 2, 81, 80, 80, 83, 5, 6, 84, 7, 8]


def suffix(*parts):
    out = []
    for p in parts:
        out.extend(p)
    return out


def test_regions_blocks_and_answer():
    s = suffix([10, 11, 12], [91, 90, 90, 92], [13, 14], [91, 90, 90, 92], [15, 70, 71, 16, 17], [84])
    ids = PROMPT + s
    lay = build_layout(ids, len(PROMPT), T, latent_size=2, control_k=2)
    assert lay.base_regions[0] == SINK
    assert [lay.base_regions[i] for i in (1, 2)] == [TEMPLATE, TEMPLATE]
    assert lay.image_positions == (3, 4)
    assert [lay.base_regions[i] for i in (6, 7)] == [QUESTION, QUESTION]
    assert lay.base_regions[8] == TEMPLATE and lay.base_regions[5] == TEMPLATE
    assert [b.start for b in lay.blocks] == [14, 20]
    assert lay.blocks[0].pads == (15, 16) and lay.blocks[0].end == 17
    assert lay.pre_text[0] == (12, 13) and lay.post_text[0] == (18, 19)
    assert lay.pre_text[1] == (18, 19)
    assert lay.post_span[1] == (24,)             # marker onward excluded
    assert lay.answer_positions == (27, 28) and lay.answer_status == "ok"
    assert latent_query_positions(lay) == (14, 15, 16, 20, 21, 22)
    assert 15 not in lay.scored_positions and 17 not in lay.scored_positions
    assert 14 in lay.scored_positions


def test_query_regions_split_current_and_previous_block():
    s = suffix([10], [91, 90, 90, 92], [13], [91, 90, 90, 92], [70, 71, 5], [84])
    lay = build_layout(PROMPT + s, len(PROMPT), T, latent_size=2, control_k=8)
    q = lay.blocks[1].pads[1]
    codes = query_regions(lay, q)
    assert len(codes) == q + 1
    assert codes[lay.blocks[0].start] == PREV_BLOCK
    assert codes[lay.blocks[1].start] == CURRENT_BLOCK and codes[q] == CURRENT_BLOCK
    assert codes[3] == IMAGE and codes[0] == SINK


def test_adjacent_blocks_and_missing_marker_are_reported_not_fatal():
    s = suffix([91, 90, 90, 92], [91, 90, 90, 92], [84])
    lay = build_layout(PROMPT + s, len(PROMPT), T, latent_size=2, control_k=8)
    assert lay.pre_text == ((), ()) and lay.post_text == ((), ())
    assert lay.post_span == ((), ())
    assert lay.answer_positions == () and lay.answer_status == "missing_marker"


def test_malformed_block_raises():
    s = suffix([91, 90, 92], [84])
    with pytest.raises(ValueError, match="latent block"):
        build_layout(PROMPT + s, len(PROMPT), T, latent_size=2)
