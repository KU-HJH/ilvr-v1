"""Token-level regions and position sets for latent image attention analysis."""
from dataclasses import dataclass

SINK, TEMPLATE, IMAGE, QUESTION, REASONING, LATENT = range(6)
PREV_BLOCK, CURRENT_BLOCK = 5, 6
REGION_NAMES = ("sink", "template", "image", "question", "reasoning", "prev_block", "current_block")


@dataclass(frozen=True)
class LayoutTokenIds:
    image: int
    vision_start: int
    vision_end: int
    im_end: int
    latent_start: int
    latent_pad: int
    latent_end: int
    answer_marker: tuple[int, ...] = (1590, 4226, 374, 25)


@dataclass(frozen=True)
class Block:
    index: int
    start: int
    pads: tuple[int, ...]
    end: int


@dataclass(frozen=True)
class Layout:
    length: int
    prompt_length: int
    base_regions: tuple[int, ...]
    block_of: tuple[int, ...]
    blocks: tuple[Block, ...]
    image_positions: tuple[int, ...]
    pre_text: tuple[tuple[int, ...], ...]
    post_text: tuple[tuple[int, ...], ...]
    post_span: tuple[tuple[int, ...], ...]
    answer_positions: tuple[int, ...]
    answer_status: str
    scored_positions: tuple[int, ...]


def _last_marker(ids, start, marker):
    n = len(marker)
    for i in range(len(ids) - n, start - 1, -1):
        if tuple(ids[i:i + n]) == marker:
            return i
    return None


def build_layout(ids, prompt_length, token_ids, *, latent_size=8, control_k=8):
    ids = [int(x) for x in ids]
    length = len(ids)
    if not 0 < prompt_length < length:
        raise ValueError("prompt_length must split a nonempty prompt and suffix")
    t = token_ids
    regions = [TEMPLATE] * length
    regions[0] = SINK
    image = tuple(i for i in range(prompt_length) if ids[i] == t.image)
    for i in image:
        regions[i] = IMAGE
    ends = [i for i in range(prompt_length) if ids[i] == t.vision_end]
    if not image or len(ends) != 1 or ends[0] < image[-1]:
        raise ValueError("prompt requires one image span followed by one vision_end")
    q_stop = next((i for i in range(ends[0] + 1, prompt_length) if ids[i] == t.im_end), None)
    if q_stop is None:
        raise ValueError("prompt question is not terminated by im_end")
    for i in range(ends[0] + 1, q_stop):
        regions[i] = QUESTION
    block_of = [-1] * length
    blocks, i = [], prompt_length
    special = {t.im_end}
    while i < length:
        tok = ids[i]
        if tok == t.latent_start:
            pads = tuple(range(i + 1, i + 1 + latent_size))
            end = i + 1 + latent_size
            if end >= length or any(ids[p] != t.latent_pad for p in pads) or ids[end] != t.latent_end:
                raise ValueError(f"malformed latent block at position {i}")
            b = Block(len(blocks), i, pads, end)
            blocks.append(b)
            for p in range(i, end + 1):
                regions[p] = LATENT
                block_of[p] = b.index
            i = end + 1
            continue
        if tok in (t.latent_pad, t.latent_end):
            raise ValueError(f"latent token outside a latent block at position {i}")
        regions[i] = TEMPLATE if tok in special else REASONING
        i += 1
    marker = _last_marker(ids, prompt_length, t.answer_marker)
    if marker is None:
        answer, status, cutoff = (), "missing_marker", length
    else:
        stop = max((j for j in range(length) if ids[j] == t.im_end), default=length)
        answer = tuple(p for p in range(marker + len(t.answer_marker), stop) if regions[p] == REASONING)
        status, cutoff = "ok", marker
    pre, post, span = [], [], []
    for b in blocks:
        lower = blocks[b.index - 1].end + 1 if b.index else prompt_length
        upper = blocks[b.index + 1].start if b.index + 1 < len(blocks) else length
        before = [p for p in range(lower, b.start) if regions[p] == REASONING]
        after = [p for p in range(b.end + 1, upper) if regions[p] == REASONING]
        pre.append(tuple(before[-control_k:]) if control_k else ())
        post.append(tuple(after[:control_k]))
        span.append(tuple(p for p in after if p < cutoff))
    scored = tuple(p for p in range(prompt_length, length)
                   if ids[p] not in (t.latent_pad, t.latent_end))
    return Layout(length, prompt_length, tuple(regions), tuple(block_of), tuple(blocks), image,
                  tuple(pre), tuple(post), tuple(span), answer, status, scored)


def latent_query_positions(layout):
    return tuple(p for b in layout.blocks for p in (b.start, *b.pads))


def query_regions(layout, query):
    if not 0 <= query < layout.length:
        raise ValueError("query outside sequence")
    own = layout.block_of[query]
    codes = []
    for key in range(query + 1):
        code = layout.base_regions[key]
        if code == LATENT:
            code = CURRENT_BLOCK if own >= 0 and layout.block_of[key] == own else PREV_BLOCK
        codes.append(code)
    return codes
