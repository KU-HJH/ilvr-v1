import pytest
from evaluation.latent_image_layout import (
    LayoutTokenIds, build_layout, query_regions, latent_query_positions,
    SINK, TEMPLATE, IMAGE, QUESTION, REASONING, LATENT, PREV_BLOCK, CURRENT_BLOCK,
    current_block_subcodes, CB_SELF, CB_CUR_START, CB_CUR_PREV, CB_OUTSIDE,
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


# ---------------------------------------------------------------------------
# R26: current-block sub-classification (SELF / CUR_START / CUR_PREV / OUTSIDE)
# ---------------------------------------------------------------------------

def _cb_layout():
    s = suffix([10], [91, 90, 90, 92], [13], [91, 90, 90, 92], [70, 71, 5], [84])
    return build_layout(PROMPT + s, len(PROMPT), T, latent_size=2, control_k=8)


def test_current_block_subcodes_start_query_is_self_only():
    # Only SELF is possible in-block for the START query: no earlier same-block key exists.
    lay = _cb_layout()
    b = lay.blocks[0]
    codes = current_block_subcodes(lay, b.start)
    assert len(codes) == b.start + 1
    assert codes[b.start] == CB_SELF
    assert CB_CUR_START not in codes and CB_CUR_PREV not in codes


def test_current_block_subcodes_pad1_is_self_and_cur_start_only():
    lay = _cb_layout()
    b = lay.blocks[0]
    codes = current_block_subcodes(lay, b.pads[0])
    assert codes[b.start] == CB_CUR_START
    assert codes[b.pads[0]] == CB_SELF
    assert CB_CUR_PREV not in codes


def test_current_block_subcodes_pad2_sees_pad1_as_cur_prev():
    lay = _cb_layout()
    b = lay.blocks[0]
    codes = current_block_subcodes(lay, b.pads[1])
    assert codes[b.start] == CB_CUR_START
    assert codes[b.pads[0]] == CB_CUR_PREV
    assert codes[b.pads[1]] == CB_SELF


def test_current_block_subcodes_end_query_sees_all_pads_as_cur_prev():
    lay = _cb_layout()
    b = lay.blocks[0]
    codes = current_block_subcodes(lay, b.end)
    assert codes[b.start] == CB_CUR_START
    assert all(codes[p] == CB_CUR_PREV for p in b.pads)
    assert codes[b.end] == CB_SELF


def test_current_block_subcodes_every_non_block_key_is_outside():
    lay = _cb_layout()
    b = lay.blocks[1]
    codes = current_block_subcodes(lay, b.end)
    own = {b.start, *b.pads, b.end}
    for key in range(b.end + 1):
        if key not in own:
            assert codes[key] == CB_OUTSIDE
    # includes keys from a different block (PREV_BLOCK region) and non-latent regions.
    assert codes[lay.blocks[0].start] == CB_OUTSIDE
    assert codes[0] == CB_OUTSIDE                  # sink


def test_current_block_subcodes_query_outside_sequence_raises():
    lay = _cb_layout()
    with pytest.raises(ValueError, match="outside sequence"):
        current_block_subcodes(lay, lay.length)
