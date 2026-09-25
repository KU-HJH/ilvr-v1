# Latent Image Attention Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CE-matched Stage 2 모델에서 중간 visual latent가 계산될 때 원본 이미지를 참조하는지 attention 측정(③), knockout(①), image 교란 patching(②), 정보 내용(B), 보조 자유 생성으로 분석하는 도구를 만들고 100문항에 실행한다.

**Architecture:** 모델 자신의 greedy 궤적(기존 TEST400 출력)을 teacher-force한다. 모든 개입은 모델의 `stage2_rollout_engine`을 그대로 쓰고 decoder 호출만 감싸 4D visibility mask를 주입한다(09-10 E3와 같은 패턴). ②의 KV 조합은 교란 이미지 token 사본을 같은 position id로 원본 이미지 뒤에 끼워 넣고, query 종류별로 어느 사본을 볼지 mask로 정해 구현한다(image token은 prefix와 자기 이미지만 보므로 사본의 K/V는 교란 run의 image K/V와 같다). ③은 최종 forward에서 attention 입력을 hook으로 받아 필요한 query 행만 softmax를 직접 계산한다.

**Tech Stack:** Python 3.11, PyTorch 2.7.1 (CUDA 12.8), repo 내 patched Transformers (`transformers/src`), PEFT DoRA, Pillow, NumPy, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-latent-image-attention-design.md`

## Global Constraints

- 작업 폴더: `/root/workspace/ILVR/.worktrees/latent-image-attention-20260925` (branch `feat/latent-image-attention-20260925`). 모든 명령은 이 폴더에서 실행한다.
- Python은 `./venv/ilvr-venv/bin/python`만 사용한다. 테스트는 `./venv/ilvr-venv/bin/python -m pytest`.
- 모델: `checkpoints/evaluation-staging/stage2-ce-matched-gb16-s212-20260919-155759/prepared-checkpoint-212`, base: `checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct`.
- 데이터: `data/TEST.jsonl` SHA256 `6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`.
- 표본: `/root/workspace/ILVR/results/server-comparison-20260912/analysis_completed/visual-causal-panel-20260910t115000z/ce/no_latent_execution/metadata.json` SHA256 `fb1f15fef5339abde6ba74d140e5f9a1f8823cedafe8aef1d4bb245a8e0286aa`, `selection.requested_indices` 100개.
- 궤적: `/root/workspace/ILVR/.worktrees/voff-reproduction-20260917/results/eval-stage2-ce-matched-test400-20260919-155759/merged.jsonl` SHA256 `76380b6e01bc6cf7e6bd44c551cf99a8ed18b0745dbd1314c8378fe2af9396d5`.
- latent size 8, control text k=8, head 집합 H = 상위 5% (29개), blur σ = 짧은 변의 0.08, B 후보 N=5 (원본+swap 4), 중간 layer 18, bootstrap 10,000회 seed 42, 모든 난수 seed 42 기반.
- teacher-forced 분석(③①②B)은 `evaluation.vision_block_runtime.numerical_policy()` (SDPA MATH, TF32 off) 안에서 실행한다. 보조 자유 생성은 기존 eval과 같은 수치 설정(`eval.load_evaluation_artifacts`, TF32 on)을 쓴다.
- 결과는 `results/latent-image-attention-<ts>/`, 로그는 `logs/latent-image-attention-<ts>/`. 모델 weight/adapter/optimizer 상태를 `results/`에 만들지 않는다. 교란 이미지를 `data/`에 저장하지 않는다.
- 기존 모듈(`vision_block_*`, `continuous_latent.py`, `eval.py`)의 기본 동작을 바꾸지 않는다. `continuous_latent.py`에는 기본값 `None`인 선택 인자만 추가한다.
- GPU 실행은 프로세스당 GPU 1장 (`CUDA_VISIBLE_DEVICES` 하나). inference 분석이라 multinode 학습 정책 대상이 아니다.
- CPU 테스트는 GPU를 요구하지 않는다. 모델이 필요한 CPU 테스트는 `step_preference.equivalence._tiny_model()`을 쓴다.

## Review Focus

- 궤적 prompt가 eval 당시와 다르게 재구성되는 경우(이미지 경로, processor drift) → `prompt_identity` 불일치로 즉시 실패해야 한다. Task 2 테스트.
- block 사이 text가 0개이거나 마지막 block 뒤에 text가 없는 경우 → control/post 집합이 비어도 실패하지 않고 해당 지표만 빠진다. Task 1 테스트.
- 출력에 "final answer is:" marker가 없는 행(400개 중 1개) → answer 지표만 제외 사유와 함께 빠진다. Task 1 테스트.
- swap 이미지를 resize했는데 processor grid가 원본과 달라지는 경우 → 조용히 위치가 어긋나지 않고 예외가 나야 한다. Task 3 테스트.
- random knockout에서 이미지 token 수보다 eligible key가 적은 경우 → 가능한 만큼 차단하고 matched 비율을 기록한다. Task 4 테스트.

---

## 파일 구조

| 파일 | 책임 |
|---|---|
| `src/evaluation/latent_image_layout.py` | token id 열에서 region 코드, block, control/post/answer 위치 계산 (순수 함수) |
| `src/evaluation/latent_image_trajectory.py` | TEST 행 + eval 행 → 검증된 `Trajectory` (processor tensor + layout) |
| `src/evaluation/latent_image_corruptions.py` | blur/mean/swap 이미지 생성, swap 후보 선택, grid 검증 |
| `src/evaluation/latent_image_interventions.py` | physical layout, visibility mask, decoder wrapper, `run_condition` |
| `src/evaluation/latent_image_attention.py` | ③ query spec, attention 재계산, value-weighted region 집계, G1/G2 |
| `src/evaluation/latent_image_stats.py` | cosine 거리, cluster bootstrap, McNemar, 식별 정확도 |
| `src/evaluation/visual_interventions.py` | 자유 생성 E3 mask (ours1-visual-causal에서 필요한 부분만 이식) |
| `src/evaluation/continuous_latent.py` | `generate(..., intervention=None)` 선택 인자 추가 |
| `src/evaluation/latent_image_cli.py` | 단계별 실행: prepare / smoke / attention / heads / interventions / freegen / report |
| `src/evaluation/latent_image_report.py` | summary.json, report.md, 그림 |
| `scripts/analyze_latent_image.py` | CLI 진입점 (5줄) |
| `tests/evaluation/test_latent_image_*.py` | 각 모듈 CPU 테스트 |

---

### Task 1: Layout — region 코드와 위치 집합

**Files:**
- Create: `src/evaluation/latent_image_layout.py`
- Test: `tests/evaluation/test_latent_image_layout.py`

**Interfaces:**
- Produces:
  - 상수 `SINK, TEMPLATE, IMAGE, QUESTION, REASONING, LATENT = 0..5`, `PREV_BLOCK = 5`, `CURRENT_BLOCK = 6`, `REGION_NAMES: tuple[str, ...]` (7개).
  - `@dataclass(frozen=True) LayoutTokenIds(image:int, vision_start:int, vision_end:int, im_end:int, latent_start:int, latent_pad:int, latent_end:int, answer_marker: tuple[int, ...] = (1590, 4226, 374, 25))`
  - `@dataclass(frozen=True) Block(index:int, start:int, pads:tuple[int, ...], end:int)`
  - `@dataclass(frozen=True) Layout(length:int, prompt_length:int, base_regions:tuple[int, ...], block_of:tuple[int, ...], blocks:tuple[Block, ...], image_positions:tuple[int, ...], pre_text:tuple[tuple[int, ...], ...], post_text:tuple[tuple[int, ...], ...], post_span:tuple[tuple[int, ...], ...], answer_positions:tuple[int, ...], answer_status:str, scored_positions:tuple[int, ...])`
  - `build_layout(ids: list[int], prompt_length: int, token_ids: LayoutTokenIds, *, latent_size: int = 8, control_k: int = 8) -> Layout`
  - `query_regions(layout: Layout, query: int) -> list[int]` (길이 query+1, key별 코드 0..6)
  - `latent_query_positions(layout: Layout) -> tuple[int, ...]` (각 block의 START + pad 8개)

정의 (spec §3):
- position 0 → SINK. prompt 안에서 `<|image_pad|>` → IMAGE. `vision_end` 다음부터 그 뒤 첫 `im_end` 직전까지 → QUESTION. 나머지 prompt → TEMPLATE.
- suffix: block(START..END) → LATENT, `im_end`/그 외 특수 종결 token → TEMPLATE, 나머지 → REASONING.
- `pre_text[b]`: START 직전, 이전 block END(또는 prompt 끝) 이후의 REASONING 위치 중 마지막 k개.
- `post_text[b]`: END 직후, 다음 block START 이전의 REASONING 위치 중 처음 k개. `post_span[b]`: 같은 구간 전체(answer marker 이후 위치는 제외).
- `answer_positions`: suffix에서 answer marker 마지막 출현 뒤부터 마지막 `im_end` 직전까지. marker가 없으면 `()`와 `answer_status="missing_marker"`, 있으면 `"ok"`.
- `scored_positions`: suffix에서 token이 PAD, END가 아닌 모든 위치(START 포함). G5와 NLL 계산 대상.
- `query_regions`: key ≤ query만. LATENT key는 query와 같은 block이면 CURRENT_BLOCK, 아니면 PREV_BLOCK.

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_latent_image_layout.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_layout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluation.latent_image_layout'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/evaluation/latent_image_layout.py
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
```

`post_span` excludes the marker tokens themselves and everything after them.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_layout.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/latent_image_layout.py tests/evaluation/test_latent_image_layout.py
git commit -m "feat: map latent image analysis regions and positions"
```

---

### Task 2: Trajectory — eval 출력에서 검증된 teacher-forcing 입력 만들기

**Files:**
- Create: `src/evaluation/latent_image_trajectory.py`
- Test: `tests/evaluation/test_latent_image_trajectory.py`

**Interfaces:**
- Consumes: `build_layout`, `LayoutTokenIds` (Task 1); `evaluation.runtime.prepare_evaluation_input`; `multinode_evaluation.resolve_evaluation_image_paths`; `eval.open_images`.
- Produces:
  - `@dataclass(frozen=True) Trajectory(source_index:int, category:str, image_paths:tuple[str, ...], images:tuple, input_ids:torch.LongTensor, attention_mask:torch.LongTensor, pixel_values:torch.Tensor, image_grid_thw:torch.LongTensor, prompt_length:int, layout:Layout, eval_match:bool, prompt_identity:str)` — tensors on CPU, `input_ids` shape `[1, L]` = prompt + generated suffix.
  - `layout_token_ids(processor, qwen_config) -> LayoutTokenIds`
  - `load_eval_rows(path: Path) -> dict[int, dict]`
  - `load_selection(path: Path) -> tuple[int, ...]`
  - `build_trajectory(processor, test_row: dict, eval_row: dict, source_index: int, data_root: Path, token_ids: LayoutTokenIds) -> Trajectory`

동작: `prepare_evaluation_input`로 prompt tensor를 만들고 `prompt_identity`, `prompt_token_length`가 eval 행과 다르면 `ValueError("prompt identity mismatch ...")`. suffix를 붙인 `input_ids`, 같은 길이 `attention_mask`(1), layout 생성. category는 첫 이미지 경로의 두 번째 경로 성분(`images_comt/<category>/...`).

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_latent_image_trajectory.py
import json
import pytest
import torch
from evaluation import latent_image_trajectory as lt
from evaluation.latent_image_layout import LayoutTokenIds

T = LayoutTokenIds(80, 81, 83, 84, 91, 90, 92, (70, 71))


class FakePrepared:
    def __init__(self, identity):
        ids = torch.tensor([[1, 2, 81, 80, 80, 83, 5, 6, 84, 7, 8]])
        self.tensors = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                        "pixel_values": torch.zeros(2, 12), "image_grid_thw": torch.tensor([[1, 1, 2]])}
        self.prompt_token_length = ids.shape[1]
        self.prompt_identity = identity


@pytest.fixture
def patched(monkeypatch, tmp_path):
    (tmp_path / "images_comt" / "deletion").mkdir(parents=True)
    monkeypatch.setattr(lt, "_open_images", lambda row, root: (["img"], ("images_comt/deletion/1.png",)))
    monkeypatch.setattr(lt, "prepare_evaluation_input", lambda p, s, i: FakePrepared("abc"))
    return tmp_path


def eval_row(identity="abc", length=11):
    return {"index": 7, "prompt_identity": identity, "prompt_token_length": length,
            "generated_suffix_token_ids": [10, 91, 90, 90, 92, 70, 71, 5, 84], "match": True}


def test_builds_prompt_plus_suffix(patched):
    tr = lt.build_trajectory(None, {"image_input": ["x"]}, eval_row(), 7, patched, T,
                             latent_size=2)
    assert tr.input_ids.shape == (1, 20) and tr.prompt_length == 11
    assert tr.layout.blocks[0].start == 12 and tr.category == "deletion" and tr.eval_match


def test_identity_mismatch_is_fatal(patched):
    with pytest.raises(ValueError, match="prompt identity mismatch"):
        lt.build_trajectory(None, {"image_input": ["x"]}, eval_row(identity="zzz"), 7, patched, T,
                            latent_size=2)


def test_length_mismatch_is_fatal(patched):
    with pytest.raises(ValueError, match="prompt identity mismatch"):
        lt.build_trajectory(None, {"image_input": ["x"]}, eval_row(length=10), 7, patched, T,
                            latent_size=2)


def test_selection_and_eval_rows(tmp_path):
    sel = tmp_path / "m.json"
    sel.write_text(json.dumps({"selection": {"requested_indices": [3, 1], "requested_count": 2}}))
    assert lt.load_selection(sel) == (3, 1)
    rows = tmp_path / "e.jsonl"
    rows.write_text("\n".join(json.dumps({"index": i}) for i in (1, 3)))
    assert set(lt.load_eval_rows(rows)) == {1, 3}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_trajectory.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/evaluation/latent_image_trajectory.py
"""Rebuild the model's own greedy TEST trajectory as a verified teacher-forcing input."""
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from evaluation.latent_image_layout import Layout, LayoutTokenIds, build_layout
from evaluation.runtime import prepare_evaluation_input


@dataclass(frozen=True)
class Trajectory:
    source_index: int
    category: str
    image_paths: tuple
    images: tuple
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    prompt_length: int
    layout: Layout
    eval_match: bool
    prompt_identity: str


def layout_token_ids(processor, config):
    tok = processor.tokenizer
    return LayoutTokenIds(
        image=config.image_token_id, vision_start=config.vision_start_token_id,
        vision_end=config.vision_end_token_id, im_end=tok.convert_tokens_to_ids("<|im_end|>"),
        latent_start=config.latent_start_token_id, latent_pad=config.latent_pad_token_id,
        latent_end=config.latent_end_token_id)


def load_eval_rows(path):
    rows = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["index"])] = row
    return rows


def load_selection(path):
    return tuple(int(i) for i in json.loads(Path(path).read_text())["selection"]["requested_indices"])


def _open_images(test_row, data_root):
    from eval import open_images
    from multinode_evaluation import resolve_evaluation_image_paths
    paths = resolve_evaluation_image_paths(test_row["image_input"], Path(data_root).resolve())
    relative = tuple(str(Path(p).resolve().relative_to(Path(data_root).resolve())) for p in paths)
    return open_images(paths), relative


def build_trajectory(processor, test_row, eval_row, source_index, data_root, token_ids, *, latent_size=8):
    if int(eval_row["index"]) != source_index:
        raise ValueError("eval row index differs from source index")
    images, relative = _open_images(test_row, data_root)
    prepared = prepare_evaluation_input(processor, test_row, images)
    if (prepared.prompt_identity != eval_row["prompt_identity"]
            or prepared.prompt_token_length != eval_row["prompt_token_length"]):
        raise ValueError(f"prompt identity mismatch for source {source_index}")
    prompt = prepared.tensors["input_ids"]
    suffix = torch.tensor([eval_row["generated_suffix_token_ids"]], dtype=prompt.dtype)
    ids = torch.cat((prompt, suffix), dim=1)
    layout = build_layout(ids[0].tolist(), prompt.shape[1], token_ids, latent_size=latent_size)
    category = Path(relative[0]).parts[1]
    return Trajectory(source_index, category, relative, tuple(images), ids, torch.ones_like(ids),
                      prepared.tensors["pixel_values"], prepared.tensors["image_grid_thw"],
                      prompt.shape[1], layout, bool(eval_row["match"]), prepared.prompt_identity)
```

Test fixture note: the test calls `build_trajectory(..., latent_size=2)`; keep the keyword.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_trajectory.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/latent_image_trajectory.py tests/evaluation/test_latent_image_trajectory.py
git commit -m "feat: rebuild verified greedy trajectories for latent image analysis"
```

---

### Task 3: 교란 이미지와 swap 후보

**Files:**
- Create: `src/evaluation/latent_image_corruptions.py`
- Test: `tests/evaluation/test_latent_image_corruptions.py`

**Interfaces:**
- Produces:
  - `processed_size(image, image_processor) -> tuple[int, int]` — `(height, width)`; `transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize(h, w, factor=patch_size*merge_size, min_pixels=image_processor.min_pixels, max_pixels=image_processor.max_pixels)`.
  - `blur_image(image, sigma_fraction=0.08) -> PIL.Image` (`ImageFilter.GaussianBlur(radius=sigma_fraction*min(w, h))`)
  - `mean_image(image) -> PIL.Image` (채널 평균 단색, 원본 크기)
  - `resized_swap(source, target, image_processor) -> PIL.Image` (source를 target의 processed `(w, h)`로 bicubic resize)
  - `select_swaps(source_index:int, category:str, pool: dict[str, tuple[int, ...]], count:int=4, seed:int=42) -> tuple[int, ...]` — 같은 category에서 자기 자신 제외, `numpy.random.default_rng([seed, source_index])`로 비복원 추출.
  - `pixel_inputs(processor, image) -> tuple[torch.Tensor, torch.Tensor]` — `processor.image_processor(images=[image], return_tensors="pt")`의 `(pixel_values, image_grid_thw)`.
  - `require_same_grid(expected: torch.Tensor, actual: torch.Tensor, label: str) -> None` — 다르면 `ValueError(f"{label} grid differs ...")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_latent_image_corruptions.py
import numpy as np
import pytest
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from evaluation import latent_image_corruptions as lc


@pytest.fixture
def proc():
    return Qwen2VLImageProcessor(min_pixels=56 * 56, max_pixels=28 * 28 * 1280)


def img(w, h, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_blur_and_mean_preserve_size_and_are_deterministic():
    a = img(301, 187)
    assert lc.blur_image(a).size == a.size
    assert np.array_equal(np.asarray(lc.blur_image(a)), np.asarray(lc.blur_image(a)))
    m = np.asarray(lc.mean_image(a))
    assert m.shape == (187, 301, 3) and (m == m[0, 0]).all()


def test_resized_swap_matches_target_grid(proc):
    target, source = img(301, 187), img(640, 480, 1)
    swapped = lc.resized_swap(source, target, proc)
    _, g_target = lc.pixel_inputs_from(proc, target)
    _, g_swap = lc.pixel_inputs_from(proc, swapped)
    lc.require_same_grid(g_target, g_swap, "swap")


def test_grid_mismatch_raises():
    with pytest.raises(ValueError, match="swap grid differs"):
        lc.require_same_grid(torch.tensor([[1, 4, 4]]), torch.tensor([[1, 4, 6]]), "swap")


def test_select_swaps_same_category_deterministic():
    pool = {"deletion": (1, 2, 3, 4, 5, 6), "update": (7, 8)}
    a = lc.select_swaps(3, "deletion", pool)
    assert a == lc.select_swaps(3, "deletion", pool)
    assert len(a) == 4 and 3 not in a and set(a) <= set(pool["deletion"])
    with pytest.raises(ValueError, match="swap candidates"):
        lc.select_swaps(7, "update", pool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_corruptions.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/evaluation/latent_image_corruptions.py
"""Deterministic same-grid image corruptions for latent image patching."""
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


def processed_size(image, image_processor):
    w, h = image.size
    factor = image_processor.patch_size * image_processor.merge_size
    return smart_resize(h, w, factor=factor, min_pixels=image_processor.min_pixels,
                        max_pixels=image_processor.max_pixels)


def blur_image(image, sigma_fraction=0.08):
    return image.filter(ImageFilter.GaussianBlur(radius=sigma_fraction * min(image.size)))


def mean_image(image):
    mean = tuple(int(round(v)) for v in np.asarray(image, dtype=np.float64).reshape(-1, 3).mean(0))
    return Image.new("RGB", image.size, mean)


def resized_swap(source, target, image_processor):
    h, w = processed_size(target, image_processor)
    return source.resize((w, h), Image.BICUBIC)


def select_swaps(source_index, category, pool, count=4, seed=42):
    candidates = [i for i in pool.get(category, ()) if i != source_index]
    if len(candidates) < count:
        raise ValueError(f"not enough swap candidates for source {source_index} in {category}")
    rng = np.random.default_rng([seed, source_index])
    return tuple(int(i) for i in rng.choice(sorted(candidates), size=count, replace=False))


def pixel_inputs_from(image_processor, image):
    out = image_processor(images=[image], return_tensors="pt")
    return out["pixel_values"], out["image_grid_thw"]


def pixel_inputs(processor, image):
    return pixel_inputs_from(processor.image_processor, image)


def require_same_grid(expected, actual, label):
    if not torch.equal(torch.as_tensor(expected).cpu(), torch.as_tensor(actual).cpu()):
        raise ValueError(f"{label} grid differs: {expected.tolist()} vs {actual.tolist()}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_corruptions.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/latent_image_corruptions.py tests/evaluation/test_latent_image_corruptions.py
git commit -m "feat: add same-grid image corruptions for latent patching"
```

---

### Task 4: Intervention engine — visibility, 이중 이미지 layout, `run_condition`

**Files:**
- Create: `src/evaluation/latent_image_interventions.py`
- Test: `tests/evaluation/test_latent_image_interventions.py`

**Interfaces:**
- Consumes: `Layout`, `latent_query_positions` (Task 1); `Trajectory` (Task 2); `evaluation.vision_block_runtime._scores`; `stage2.peft.resolve_stage2_qwen_model`; `transformers.models.qwen2_5_vl.stage2_latent_rollout.parse_latent_spans, validate_stage2_config`.
- Produces:
  - `@dataclass(frozen=True) Condition(name:str, text_source:str="clean", latent_source:str="clean", head_subset:tuple[tuple[int, int], ...]=(), random_keys:bool=False, native:bool=False)`; source ∈ `{"clean", "alt", "none"}`. `native=True`는 모델 원래 2D mask 경로(= 생성 경로, G5 기준).
  - `@dataclass(frozen=True) PhysicalLayout(length:int, logical_of:tuple[int, ...], physical_of:tuple[int, ...], clean_image:tuple[int, ...], alt_image:tuple[int, ...], latent_rows:tuple[int, ...], block_rows:tuple[tuple[int, ...], ...])`
  - `build_physical_layout(layout: Layout, with_alt: bool) -> PhysicalLayout` — alt 사본은 마지막 원본 image 위치 바로 뒤에 삽입.
  - `build_visibility(phys: PhysicalLayout, condition: Condition, random_blocked: dict[int, tuple[int, ...]] | None = None) -> torch.BoolTensor` `[L, L]` (physical 좌표; `random_blocked`의 key는 block index, 값은 logical key 위치).
  - `random_key_sets(layout: Layout, seed: int = 42, source_index: int = 0) -> tuple[dict[int, tuple[int, ...]], dict[int, float]]` — block별 차단 key와 matched 비율(`len(chosen)/len(image)`).
  - `@dataclass ConditionResult(name:str, latents:torch.Tensor, mid_body:torch.Tensor, final_body:torch.Tensor, image_mid:torch.Tensor, image_final:torch.Tensor, last_hidden:dict[int, torch.Tensor], scores:dict[int, tuple[int, float]])` — `latents` `[n_blocks, latent_size, hidden]` fp32 CPU; `mid_body`/`final_body` `[n_blocks, hidden]` (pad 위치 평균, layer 18 / 마지막); `image_mid`/`image_final` `[hidden]` (원본 쪽 image 위치 평균; alt 조건이면 사람이 보는 쪽 = 해당 query가 보는 사본이 아닌, 텍스트 source 사본); `last_hidden`는 control text 위치(logical)→최종 hidden; `scores`는 logical scored position → (argmax, nll).
  - `run_condition(model, trajectory, condition, *, alt_pixel_values=None, alt_grid=None, pixel_override=None, random_blocked=None, mid_layer=18, final_observer=None) -> ConditionResult`
    - `pixel_override`: 원본 대신 다른 이미지 한 장으로 전체 run (B의 swap 후보). `alt_pixel_values`와 동시 사용 금지.
    - `final_observer`: 최종 forward 동안에만 들어가는 context manager factory (`Task 5`가 사용).

Visibility 규칙 (physical 좌표, `allowed = tril`):
1. alt 사본 행: 원본 image 열 차단.
2. 마지막 alt 위치 이후 행: source = `latent_source` (행이 `latent_rows`에 있을 때) 아니면 `text_source`. `clean` → alt 열 차단, `alt` → 원본 image 열 차단, `none` → 둘 다 차단.
3. alt 없는 layout에서 `alt` source는 `ValueError`.
4. `head_subset`: base는 rule 1–2 (`latent_source`를 `clean`으로 간주), layer별 pre-hook이 해당 head만 `latent_source="none"` visibility로 교체한 `[1, H, P, P]` mask를 넣는다. 들어온 mask가 base와 다르면 `ValueError("decoder altered visibility")`.
5. `random_keys`: block b의 latent 행에서 `random_blocked[b]` key 차단 (image는 보임).

`random_key_sets`: block b마다 eligible = START 이전 key 중 region이 QUESTION, REASONING, 또는 이전 block(LATENT)인 위치. 개수 = `min(len(image), len(eligible))`. `numpy.random.default_rng([seed, source_index, b])`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/evaluation/test_latent_image_interventions.py
import pytest
import torch
from step_preference.equivalence import _tiny_model
from evaluation.latent_image_layout import LayoutTokenIds, build_layout
from evaluation.latent_image_trajectory import Trajectory
from evaluation import latent_image_interventions as li

T = LayoutTokenIds(80, 81, 83, 84, 91, 90, 92, (70, 71))
PROMPT = [1, 2, 81, 80, 80, 80, 80, 83, 5, 6, 84, 7, 8]
SUFFIX = [10, 11, 91, 90, 90, 92, 13, 14, 91, 90, 90, 92, 15, 70, 71, 16, 84]


def layout():
    return build_layout(PROMPT + SUFFIX, len(PROMPT), T, latent_size=2, control_k=2)


@pytest.fixture(scope="module")
def tiny():
    # Same construction as tests/evaluation/test_vision_block_runtime.py::image_case.
    torch.manual_seed(0)
    config = _tiny_model().config
    config.num_hidden_layers = 2
    config.vision_end_token_id = 83
    config._attn_implementation = "sdpa"
    return type(_tiny_model())(config).eval()


def trajectory(pixel_seed=0):
    ids = torch.tensor([PROMPT + SUFFIX])
    g = torch.Generator().manual_seed(pixel_seed)
    return Trajectory(0, "deletion", ("x",), (), ids, torch.ones_like(ids),
                      torch.randn(4, 12, generator=g), torch.tensor([[1, 2, 2]]), len(PROMPT),
                      layout(), True, "id")


def test_physical_layout_inserts_alt_after_image():
    phys = li.build_physical_layout(layout(), with_alt=True)
    assert phys.clean_image == (3, 4, 5, 6) and phys.alt_image == (7, 8, 9, 10)
    assert phys.length == len(PROMPT + SUFFIX) + 4
    assert phys.physical_of[11] == 15 and phys.logical_of[8] == 4


def test_visibility_routes_sources():
    phys = li.build_physical_layout(layout(), with_alt=True)
    direct = li.build_visibility(phys, li.Condition("direct", "clean", "alt"))
    latent_row, text_row = phys.latent_rows[0], phys.physical_of[layout().pre_text[0][0]]
    assert not direct[latent_row, list(phys.clean_image)].any() and direct[latent_row, list(phys.alt_image)].all()
    assert direct[text_row, list(phys.clean_image)].all() and not direct[text_row, list(phys.alt_image)].any()
    assert not direct[phys.alt_image[-1], list(phys.clean_image)].any()
    with pytest.raises(ValueError, match="alt"):
        li.build_visibility(li.build_physical_layout(layout(), False), li.Condition("x", "alt", "alt"))


def test_random_keys_before_block_and_deterministic():
    sets, matched = li.random_key_sets(layout(), seed=42, source_index=0)
    assert set(sets) == {0, 1}
    assert all(k < layout().blocks[b].start for b, keys in sets.items() for k in keys)
    assert matched == {0: 1.0, 1: 1.0}
    assert sets == li.random_key_sets(layout(), seed=42, source_index=0)[0]


def test_random_keys_capped_when_eligible_is_short():
    # Block right after the prompt: only the two question tokens are eligible for 4 image tokens.
    short = build_layout(PROMPT + [91, 90, 90, 92, 70, 71, 84], len(PROMPT), T, latent_size=2)
    sets, matched = li.random_key_sets(short, seed=42, source_index=0)
    assert sets[0] == (8, 9) and matched[0] == pytest.approx(0.5)


def test_noop_matches_native_and_dual_clean_matches_single(tiny):
    tr = trajectory()
    native = li.run_condition(tiny, tr, li.Condition("reference", native=True), mid_layer=0)
    noop = li.run_condition(tiny, tr, li.Condition("noop"), mid_layer=0)
    alt_pixels = torch.randn(4, 12, generator=torch.Generator().manual_seed(5))
    dual = li.run_condition(tiny, tr, li.Condition("dual_clean"), alt_pixel_values=alt_pixels,
                            alt_grid=tr.image_grid_thw, mid_layer=0)
    for other in (noop, dual):
        torch.testing.assert_close(other.latents, native.latents, rtol=1e-4, atol=1e-5)
        assert {k: v[0] for k, v in other.scores.items()} == {k: v[0] for k, v in native.scores.items()}


def test_total_alt_equals_override_and_knockout_changes_latents(tiny):
    tr = trajectory()
    alt_pixels = torch.randn(4, 12, generator=torch.Generator().manual_seed(5))
    total = li.run_condition(tiny, tr, li.Condition("total", "alt", "alt"), alt_pixel_values=alt_pixels,
                             alt_grid=tr.image_grid_thw, mid_layer=0)
    override = li.run_condition(tiny, tr, li.Condition("swap"), pixel_override=(alt_pixels, tr.image_grid_thw),
                                mid_layer=0)
    torch.testing.assert_close(total.latents, override.latents, rtol=1e-4, atol=1e-5)
    noop = li.run_condition(tiny, tr, li.Condition("noop"), mid_layer=0)
    k_all = li.run_condition(tiny, tr, li.Condition("k_all", latent_source="none"), mid_layer=0)
    assert not torch.allclose(k_all.latents, noop.latents)


def test_head_subset_all_heads_equals_k_all(tiny):
    tr = trajectory()
    heads = tuple((layer, h) for layer in range(tiny.config.num_hidden_layers)
                  for h in range(tiny.config.num_attention_heads))
    subset = li.run_condition(tiny, tr, li.Condition("k_heads", head_subset=heads), mid_layer=0)
    k_all = li.run_condition(tiny, tr, li.Condition("k_all", latent_source="none"), mid_layer=0)
    torch.testing.assert_close(subset.latents, k_all.latents, rtol=1e-4, atol=1e-5)


def test_head_subset_single_head_is_between(tiny):
    tr = trajectory()
    noop = li.run_condition(tiny, tr, li.Condition("noop"), mid_layer=0)
    one = li.run_condition(tiny, tr, li.Condition("k_one", head_subset=((0, 1),)), mid_layer=0)
    assert not torch.allclose(one.latents, noop.latents)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_interventions.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement layout, visibility and random keys**

```python
# src/evaluation/latent_image_interventions.py
"""Teacher-forced latent recomputation under image-routing interventions."""
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch

from evaluation.latent_image_layout import LATENT, QUESTION, REASONING, latent_query_positions

SOURCES = ("clean", "alt", "none")


@dataclass(frozen=True)
class Condition:
    name: str
    text_source: str = "clean"
    latent_source: str = "clean"
    head_subset: tuple = ()
    random_keys: bool = False
    native: bool = False

    def __post_init__(self):
        if self.text_source not in SOURCES or self.latent_source not in SOURCES:
            raise ValueError(f"unknown image source in {self}")
        if self.head_subset and self.latent_source != "clean":
            raise ValueError("head_subset requires latent_source='clean' as its base")
        if self.native and (self.text_source, self.latent_source) != ("clean", "clean"):
            raise ValueError("native condition cannot route images")


@dataclass(frozen=True)
class PhysicalLayout:
    length: int
    logical_of: tuple
    physical_of: tuple
    clean_image: tuple
    alt_image: tuple
    latent_rows: tuple
    block_rows: tuple


def build_physical_layout(layout, with_alt):
    image = layout.image_positions
    n = len(image) if with_alt else 0
    last = image[-1]
    physical_of = tuple(p if p <= last else p + n for p in range(layout.length))
    logical_of = list(range(last + 1)) + list(image[:n]) + list(range(last + 1, layout.length))
    alt = tuple(range(last + 1, last + 1 + n))
    rows = tuple(physical_of[p] for p in latent_query_positions(layout))
    block_rows = tuple(tuple(physical_of[p] for p in (b.start, *b.pads)) for b in layout.blocks)
    return PhysicalLayout(layout.length + n, tuple(logical_of), physical_of, tuple(image), alt, rows, block_rows)


def build_visibility(phys, condition, random_blocked=None, *, latent_source=None):
    latent_source = latent_source or condition.latent_source
    if not phys.alt_image and "alt" in (condition.text_source, latent_source):
        raise ValueError("alt image source requires a dual-image layout")
    allowed = torch.ones(phys.length, phys.length, dtype=torch.bool).tril()
    clean, alt = list(phys.clean_image), list(phys.alt_image)
    if alt:
        allowed[alt[0]:alt[-1] + 1, clean] = False
    first = (alt[-1] if alt else clean[-1]) + 1
    latent_rows = set(phys.latent_rows)
    for q in range(first, phys.length):
        source = latent_source if q in latent_rows else condition.text_source
        if source in ("clean", "none") and alt:
            allowed[q, alt] = False
        if source in ("alt", "none"):
            allowed[q, clean] = False
    if condition.random_keys:
        if random_blocked is None:
            raise ValueError("random_keys condition requires random_blocked")
        for block, rows in enumerate(phys.block_rows):
            keys = [phys.physical_of[k] for k in random_blocked.get(block, ())]
            if keys:
                allowed[torch.tensor(rows)[:, None], torch.tensor(keys)[None, :]] = False
    return allowed


def random_key_sets(layout, seed=42, source_index=0):
    sets, matched = {}, {}
    n_image = len(layout.image_positions)
    for b in layout.blocks:
        eligible = [k for k in range(b.start)
                    if layout.base_regions[k] in (QUESTION, REASONING, LATENT)]
        count = min(n_image, len(eligible))
        rng = np.random.default_rng([seed, source_index, b.index])
        chosen = sorted(int(k) for k in rng.choice(eligible, size=count, replace=False)) if count else []
        sets[b.index] = tuple(chosen)
        matched[b.index] = count / n_image
    return sets, matched
```

- [ ] **Step 4: Run the geometry tests**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_interventions.py -k "physical or visibility or random" -v`
Expected: 4 passed

- [ ] **Step 5: Implement `run_condition`**

```python
# append to src/evaluation/latent_image_interventions.py
from stage2.peft import resolve_stage2_qwen_model
from transformers.models.qwen2_5_vl.stage2_latent_rollout import parse_latent_spans, validate_stage2_config


@dataclass
class ConditionResult:
    name: str
    latents: torch.Tensor
    mid_body: torch.Tensor
    final_body: torch.Tensor
    image_mid: torch.Tensor
    image_final: torch.Tensor
    last_hidden: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)


def _additive(allowed, dtype):
    """[P, P] -> [1, 1, P, P]; [H, P, P] -> [1, H, P, P]."""
    mask = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    mask = mask.masked_fill(~allowed, float("-inf"))
    return mask[None, None] if allowed.ndim == 2 else mask[None]


@contextmanager
def _head_masks(qwen, base_allowed, none_allowed, head_subset, dtype):
    by_layer = {}
    for layer, head in head_subset:
        by_layer.setdefault(layer, set()).add(head)
    heads = qwen.config.num_attention_heads
    handles = []

    def hook(index):
        chosen = torch.zeros(heads, dtype=torch.bool, device=base_allowed.device)
        chosen[list(by_layer[index])] = True

        def pre(module, args, kwargs):
            incoming = kwargs["attention_mask"]
            p = incoming.shape[-1]
            if not torch.equal(torch.isfinite(incoming[0, 0]), base_allowed[:p, :p]):
                raise ValueError("decoder altered visibility")
            allowed = torch.where(chosen[:, None, None], none_allowed[None, :p, :p], base_allowed[None, :p, :p])
            kwargs["attention_mask"] = _additive(allowed, incoming.dtype)      # [1, H, P, P]
            return args, kwargs
        return pre

    try:
        for index in by_layer:
            handles.append(qwen.model.layers[index].self_attn.register_forward_pre_hook(hook(index), with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _physical_embeds(qwen, trajectory, phys, alt_pixel_values, alt_grid, pixel_override):
    device = qwen.model.embed_tokens.weight.device
    ids = trajectory.input_ids.to(device)
    pixels, grid = (pixel_override if pixel_override is not None
                    else (trajectory.pixel_values, trajectory.image_grid_thw))
    grid = grid.to(device)
    if pixel_override is not None and not torch.equal(grid.cpu(), trajectory.image_grid_thw):
        raise ValueError("override grid differs from trajectory grid")
    base = qwen.prepare_stage2_base_inputs_embeds(input_ids=ids, pixel_values=pixels.to(device), image_grid_thw=grid)
    positions, _ = qwen.get_rope_index(ids, image_grid_thw=grid, attention_mask=trajectory.attention_mask.to(device))
    index = torch.tensor(phys.logical_of, device=device)
    embeds = base.index_select(1, index)
    physical_ids = ids.index_select(1, index)
    if phys.alt_image:
        if alt_grid is None or not torch.equal(alt_grid.cpu(), trajectory.image_grid_thw):
            raise ValueError("alt grid differs from trajectory grid")
        with torch.no_grad():
            alt = qwen.visual(alt_pixel_values.to(device, qwen.visual.dtype), grid_thw=alt_grid.to(device))
        embeds[:, list(phys.alt_image)] = alt.to(embeds.dtype)[None]
    return embeds, positions.index_select(2, index), physical_ids


@torch.inference_mode()
def run_condition(model, trajectory, condition, *, alt_pixel_values=None, alt_grid=None, pixel_override=None,
                  random_blocked=None, mid_layer=18, final_observer=None):
    from evaluation.vision_block_runtime import _scores
    if alt_pixel_values is not None and pixel_override is not None:
        raise ValueError("alt image and pixel override are exclusive")
    qwen = resolve_stage2_qwen_model(model)
    layout = trajectory.layout
    phys = build_physical_layout(layout, with_alt=alt_pixel_values is not None)
    embeds, positions, physical_ids = _physical_embeds(qwen, trajectory, phys, alt_pixel_values, alt_grid, pixel_override)
    device, dtype = embeds.device, embeds.dtype
    token_ids, latent_size = validate_stage2_config(qwen.config)
    spans = parse_latent_spans(physical_ids[0], token_ids=token_ids, latent_size=latent_size)
    if condition.native:
        allowed = None
    else:
        allowed = build_visibility(phys, condition, random_blocked).to(device)
    stack = ExitStack()
    if condition.head_subset:
        none_allowed = build_visibility(phys, condition, random_blocked, latent_source="none").to(device)
        stack.enter_context(_head_masks(qwen, allowed, none_allowed, condition.head_subset, dtype))

    def decoder(**kwargs):
        if allowed is not None:
            p = kwargs["inputs_embeds"].shape[1]
            kwargs["attention_mask"] = _additive(allowed[:p, :p], dtype)
        return qwen.model(generate_mode=True, **kwargs)

    with stack:
        rollout = qwen.stage2_rollout_engine.rollout(
            language_model=decoder, base_inputs_embeds=embeds,
            attention_mask=torch.ones(physical_ids.shape, dtype=torch.long, device=device),
            full_position_ids=positions, latent_spans=spans)
        with (final_observer() if final_observer else nullcontext()):
            final = decoder(inputs_embeds=rollout.final_inputs_embeds,
                            attention_mask=torch.ones(physical_ids.shape, dtype=torch.long, device=device),
                            position_ids=positions, use_cache=False, return_dict=True, output_hidden_states=True)
    hidden = final.last_hidden_state
    n_blocks = len(layout.blocks)
    latents = torch.stack([s[0] for s in rollout.latent_states]).float().reshape(n_blocks, latent_size, -1).cpu()
    pad_rows = [[phys.physical_of[p] for p in b.pads] for b in layout.blocks]
    mid = final.hidden_states[mid_layer][0]
    mid_body = torch.stack([mid[r].float().mean(0) for r in pad_rows]).cpu()
    final_body = torch.stack([hidden[0, r].float().mean(0) for r in pad_rows]).cpu()
    image_rows = list(phys.alt_image if condition.text_source == "alt" else phys.clean_image)
    targets = list(layout.scored_positions)
    predictors = [phys.physical_of[p] - 1 for p in targets]
    token_list = trajectory.input_ids[0].tolist()
    scores = dict(zip(targets, _scores(qwen, hidden, predictors, [token_list[p] for p in targets])))
    controls = {p for group in (layout.pre_text + layout.post_text) for p in group}
    return ConditionResult(
        condition.name, latents, mid_body, final_body,
        mid[image_rows].float().mean(0).cpu(), hidden[0, image_rows].float().mean(0).cpu(),
        {p: hidden[0, phys.physical_of[p]].float().cpu() for p in controls}, scores)
```

- [ ] **Step 6: Run all engine tests**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_interventions.py -v`
Expected: 8 passed. If `test_noop_matches_native...` fails on scores, print `max |Δnll|`; a nonzero argmax difference means the 4D mask path differs from the native path — stop and investigate (do not loosen the test).

- [ ] **Step 7: Commit**

```bash
git add src/evaluation/latent_image_interventions.py tests/evaluation/test_latent_image_interventions.py
git commit -m "feat: recompute latents under image routing interventions"
```

---

### Task 5: ③ Attention 측정

**Files:**
- Create: `src/evaluation/latent_image_attention.py`
- Test: `tests/evaluation/test_latent_image_attention.py`

**Interfaces:**
- Consumes: `Layout`, `query_regions`, `REGION_NAMES` (Task 1); `run_condition`, `Condition` (Task 4); `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb, repeat_kv`.
- Produces:
  - `@dataclass(frozen=True) QuerySpec(position:int, group:str, block:int, slot:str)`; group ∈ `{"latent", "pre_text", "post_text"}`; slot ∈ `{"start", "pad1".."pad8", "end", "pre", "post"}`.
  - `query_specs(layout) -> tuple[QuerySpec, ...]`
  - `@dataclass AttentionCapture(specs:tuple[QuerySpec, ...], mass:np.ndarray, vw_share:np.ndarray, enrichment:np.ndarray, image_probs:np.ndarray | None, g1_max_rel:float, g2_max_abs:float)` — `mass`, `vw_share` shape `[rows, layers, heads, 7]` float16; `enrichment` `[rows, layers, heads]` float32; `image_probs` `[rows, layers, image_tokens]` (head 평균, `keep_image_probs=True`일 때만).
  - `capture_attention(model, trajectory, *, keep_image_probs=False) -> AttentionCapture` — `run_condition(model, trajectory, Condition("noop"), final_observer=...)`의 최종 forward에서만 hook이 동작.

계산 (layer마다, `x` = self_attn 입력 hidden `[L, D]`, `R` = spec 위치):
1. `q, k, v` = `q_proj/k_proj/v_proj(x)` → `[1, heads|kv, L, 128]`, `apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)`.
2. `scores = q[:, :, R] @ repeat_kv(k)ᵀ / √128 + additive(mask)` (hook으로 받은 실제 `attention_mask`의 R 행), fp32 softmax → `probs [heads, |R|, L]`.
3. G1: `o_proj((probs @ repeat_kv(v)) → [|R|, heads*128])`를 self_attn 실제 출력의 R 행과 비교, 최대 상대오차 기록.
4. `contrib[h, j] = ‖o_proj(e_h ⊗ v_{g(h), j})‖` — head h 슬라이스만 채운 `[L, heads*128]` 입력을 `o_proj`에 통과(LoRA/DoRA 포함 선형). G1b: `o_proj(Σ_h X_h)`와 `Σ_h o_proj(X_h)` 일치(선형성) 확인.
5. region 집계: 행 r의 `query_regions` 코드별 `probs` 합 → `mass`; `probs*contrib` 합을 전체로 나눈 비율 → `vw_share`. G2: region 합이 1 ± 1e-3, 코드 수 = visible key 수.
6. `enrichment = mass[IMAGE] / (n_image / n_visible)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_latent_image_attention.py
import numpy as np
import torch
from evaluation import latent_image_attention as la
from evaluation.latent_image_layout import IMAGE
from tests.evaluation.test_latent_image_interventions import tiny, trajectory  # noqa: F401  (fixture reuse)


def test_query_specs_cover_blocks_and_controls():
    tr = trajectory()
    specs = la.query_specs(tr.layout)
    latent = [s for s in specs if s.group == "latent"]
    assert len(latent) == len(tr.layout.blocks) * (2 + 2)       # start, 2 pads, end (tiny latent_size=2)
    assert {s.slot for s in specs if s.group == "pre_text"} == {"pre"}
    assert all(s.position in tr.layout.post_text[s.block] for s in specs if s.group == "post_text")


def test_capture_reproduces_sdpa_and_partitions_mass(tiny):
    tr = trajectory()
    cap = la.capture_attention(tiny, tr, keep_image_probs=True)
    assert cap.g1_max_rel < 1e-3
    assert cap.g2_max_abs < 1e-3
    layers = tiny.config.num_hidden_layers
    assert cap.mass.shape[1:] == (layers, tiny.config.num_attention_heads, 7)
    np.testing.assert_allclose(cap.mass.astype(np.float32).sum(-1), 1, atol=2e-3)
    np.testing.assert_allclose(cap.vw_share.astype(np.float32).sum(-1), 1, atol=2e-3)
    assert cap.image_probs.shape == (len(cap.specs), layers, 4)
    rows = [i for i, s in enumerate(cap.specs) if s.group == "latent"]
    assert (cap.mass[rows, :, :, IMAGE] >= 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_attention.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/evaluation/latent_image_attention.py
"""Descriptive attention of latent and control queries over input regions (analysis ③)."""
from contextlib import contextmanager
from dataclasses import dataclass
import math

import numpy as np
import torch

from evaluation.latent_image_interventions import Condition, run_condition
from evaluation.latent_image_layout import IMAGE, REGION_NAMES, query_regions
from stage2.peft import resolve_stage2_qwen_model
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, repeat_kv


@dataclass(frozen=True)
class QuerySpec:
    position: int
    group: str
    block: int
    slot: str


@dataclass
class AttentionCapture:
    specs: tuple
    mass: np.ndarray
    vw_share: np.ndarray
    enrichment: np.ndarray
    image_probs: object
    g1_max_rel: float
    g2_max_abs: float


def query_specs(layout):
    specs = []
    for b in layout.blocks:
        specs.append(QuerySpec(b.start, "latent", b.index, "start"))
        specs.extend(QuerySpec(p, "latent", b.index, f"pad{i + 1}") for i, p in enumerate(b.pads))
        specs.append(QuerySpec(b.end, "latent", b.index, "end"))
        specs.extend(QuerySpec(p, "pre_text", b.index, "pre") for p in layout.pre_text[b.index])
        specs.extend(QuerySpec(p, "post_text", b.index, "post") for p in layout.post_text[b.index])
    return tuple(specs)


def capture_attention(model, trajectory, *, keep_image_probs=False):
    qwen = resolve_stage2_qwen_model(model)
    layout = trajectory.layout
    specs = query_specs(layout)
    rows = [s.position for s in specs]              # single-image layout: physical == logical
    codes = [torch.tensor(query_regions(layout, p)) for p in rows]
    n_layers, heads = len(qwen.model.layers), qwen.config.num_attention_heads
    mass = np.zeros((len(rows), n_layers, heads, len(REGION_NAMES)), np.float32)
    vw = np.zeros_like(mass)
    enrich = np.zeros((len(rows), n_layers, heads), np.float32)
    image_probs = np.zeros((len(rows), n_layers, len(layout.image_positions)), np.float32) if keep_image_probs else None
    worst = {"g1": 0.0, "g2": 0.0}

    def layer_hooks(index, attn):
        seen = {}

        def pre(module, args, kwargs):
            seen["x"] = kwargs["hidden_states"].detach()
            seen["pe"] = kwargs["position_embeddings"]
            seen["mask"] = kwargs["attention_mask"]

        def post(module, args, kwargs, output):
            x, (cos, sin), mask = seen["x"], seen["pe"], seen["mask"]
            _, length, _ = x.shape
            hd = attn.head_dim
            q = attn.q_proj(x).view(1, length, -1, hd).transpose(1, 2)
            k = attn.k_proj(x).view(1, length, -1, hd).transpose(1, 2)
            v = attn.v_proj(x).view(1, length, -1, hd).transpose(1, 2)
            q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, attn.rope_scaling["mrope_section"])
            kk, vv = repeat_kv(k, attn.num_key_value_groups), repeat_kv(v, attn.num_key_value_groups)
            r = torch.tensor(rows, device=x.device)
            scores = (q[:, :, r].float() @ kk.float().transpose(-1, -2)) / math.sqrt(hd)
            scores = scores + mask[:, :, r, :length].float()
            probs = scores.softmax(-1)[0]                                      # [H, R, L]
            out = (probs.to(vv.dtype) @ vv[0]).transpose(0, 1).reshape(len(rows), -1)
            mine = attn.o_proj(out).float()
            actual = output[0][0, r].float()
            worst["g1"] = max(worst["g1"], float(((mine - actual).norm(dim=-1) / actual.norm(dim=-1).clamp_min(1e-6)).max()))
            # One head at a time bounds memory at [L, heads*hd] for the 3B model.
            contrib = torch.empty(heads, length, device=x.device)
            total = torch.zeros(length, heads * hd, device=x.device, dtype=v.dtype)
            summed = None
            for h in range(heads):
                block = torch.zeros(length, heads * hd, device=x.device, dtype=v.dtype)
                block[:, h * hd:(h + 1) * hd] = vv[0, h]
                projected = attn.o_proj(block).float()
                contrib[h] = projected.norm(dim=-1)
                summed = projected if summed is None else summed + projected
                total += block
            linear_err = float((attn.o_proj(total).float() - summed).abs().max() / summed.abs().max().clamp_min(1e-6))
            worst["g1"] = max(worst["g1"], linear_err)
            weighted = probs * contrib[:, None, :]
            for i, code in enumerate(codes):
                code = code.to(x.device)
                p_row, w_row = probs[:, i, :code.numel()], weighted[:, i, :code.numel()]
                m = torch.zeros(heads, len(REGION_NAMES), device=x.device).index_add_(1, code, p_row)
                w = torch.zeros_like(m).index_add_(1, code, w_row)
                w = w / w.sum(-1, keepdim=True)
                worst["g2"] = max(worst["g2"], float((m.sum(-1) - 1).abs().max()),
                                  float((p_row.sum(-1) - probs[:, i].sum(-1)).abs().max()))
                mass[i, index], vw[i, index] = m.cpu().numpy(), w.cpu().numpy()
                n_img = int((code == IMAGE).sum())
                enrich[i, index] = (m[:, IMAGE] / (n_img / code.numel())).cpu().numpy()
                if image_probs is not None:
                    image_probs[i, index] = probs[:, i, list(layout.image_positions)].mean(0).cpu().numpy()
        return pre, post

    @contextmanager
    def observer():
        handles = []
        try:
            for index, layer in enumerate(qwen.model.layers):
                pre, post = layer_hooks(index, layer.self_attn)
                handles.append(layer.self_attn.register_forward_pre_hook(pre, with_kwargs=True))
                handles.append(layer.self_attn.register_forward_hook(post, with_kwargs=True))
            yield
        finally:
            for handle in handles:
                handle.remove()

    run_condition(model, trajectory, Condition("noop"), final_observer=observer, mid_layer=0)
    return AttentionCapture(specs, mass.astype(np.float16), vw.astype(np.float16), enrich, image_probs,
                            worst["g1"], worst["g2"])
```

The mask rows come from the hooked `attention_mask`, so G1 compares exactly what SDPA saw.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_attention.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/latent_image_attention.py tests/evaluation/test_latent_image_attention.py
git commit -m "feat: measure value-weighted latent attention over image regions"
```

---

### Task 6: 통계와 지표

**Files:**
- Create: `src/evaluation/latent_image_stats.py`
- Test: `tests/evaluation/test_latent_image_stats.py`

**Interfaces:**
- Produces:
  - `cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray` (마지막 축 기준, `1 - cos`)
  - `cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, n: int = 10000, seed: int = 42) -> dict` → `{"mean", "lo", "hi", "n_units", "n_clusters"}` (cluster 재표집 후 unit 평균; NaN unit 제거)
  - `paired_cluster_bootstrap(a, b, clusters, n=10000, seed=42) -> dict` (a−b)
  - `ratio_cluster_bootstrap(num, den, clusters, n=10000, seed=42) -> dict` (Σnum/Σden의 cluster bootstrap)
  - `exact_mcnemar(b01: int, b10: int) -> float` (양측 이항, 둘 다 0이면 1.0)
  - `identification_accuracy(latent: np.ndarray, image: np.ndarray) -> float` — 입력 `[questions, N, d]`. 문항 내에서 후보 평균을 빼(centering) 질문 공통 성분을 제거한 뒤, 각 후보 i의 latent가 image 표현 중 cosine 최댓값을 i에서 갖는 비율. 우연 수준 1/N.

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_latent_image_stats.py
import numpy as np
import pytest
from evaluation import latent_image_stats as st


def test_cosine_distance():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(st.cosine_distance(a, a), [0, 0], atol=1e-12)
    np.testing.assert_allclose(st.cosine_distance(a, a[::-1]), [1, 1], atol=1e-12)


def test_cluster_bootstrap_respects_clusters():
    values = np.array([1.0, 1.0, 1.0, 3.0])
    clusters = np.array([0, 0, 0, 1])
    out = st.cluster_bootstrap(values, clusters, n=2000)
    assert out["n_clusters"] == 2 and out["lo"] <= out["mean"] <= out["hi"]
    assert out["mean"] == pytest.approx(1.5)
    same = st.cluster_bootstrap(np.ones(6), np.arange(6), n=500)
    assert same["lo"] == same["hi"] == 1


def test_paired_and_ratio():
    c = np.arange(5)
    d = st.paired_cluster_bootstrap(np.full(5, 2.0), np.ones(5), c, n=200)
    assert d["mean"] == pytest.approx(1)
    r = st.ratio_cluster_bootstrap(np.full(5, 1.0), np.full(5, 4.0), c, n=200)
    assert r["mean"] == pytest.approx(0.25)


def test_mcnemar():
    assert st.exact_mcnemar(0, 0) == 1.0
    assert st.exact_mcnemar(0, 6) == pytest.approx(0.03125)


def test_identification_accuracy():
    rng = np.random.default_rng(0)
    image = rng.normal(size=(20, 5, 16))
    assert st.identification_accuracy(image + 0.01 * rng.normal(size=image.shape), image) == 1.0
    assert st.identification_accuracy(rng.normal(size=image.shape), image) < 0.6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_stats.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/evaluation/latent_image_stats.py
"""Question-clustered statistics for latent image analysis."""
from math import comb

import numpy as np


def cosine_distance(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return 1 - num / np.maximum(den, 1e-12)


def _resample(clusters, n, seed):
    ids = np.unique(clusters)
    rng = np.random.default_rng(seed)
    index = {c: np.flatnonzero(clusters == c) for c in ids}
    for _ in range(n):
        yield np.concatenate([index[c] for c in rng.choice(ids, size=len(ids), replace=True)])


def _summary(point, draws, values, clusters):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"mean": float(point), "lo": float(lo), "hi": float(hi),
            "n_units": int(len(values)), "n_clusters": int(len(np.unique(clusters)))}


def cluster_bootstrap(values, clusters, n=10000, seed=42):
    values, clusters = np.asarray(values, np.float64), np.asarray(clusters)
    keep = ~np.isnan(values)
    values, clusters = values[keep], clusters[keep]
    draws = [values[i].mean() for i in _resample(clusters, n, seed)]
    return _summary(values.mean(), draws, values, clusters)


def paired_cluster_bootstrap(a, b, clusters, n=10000, seed=42):
    return cluster_bootstrap(np.asarray(a, np.float64) - np.asarray(b, np.float64), clusters, n, seed)


def ratio_cluster_bootstrap(num, den, clusters, n=10000, seed=42):
    num, den, clusters = np.asarray(num, np.float64), np.asarray(den, np.float64), np.asarray(clusters)
    keep = ~(np.isnan(num) | np.isnan(den))
    num, den, clusters = num[keep], den[keep], clusters[keep]
    draws = [num[i].sum() / den[i].sum() for i in _resample(clusters, n, seed)]
    return _summary(num.sum() / den.sum(), draws, num, clusters)


def exact_mcnemar(b01, b10):
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def identification_accuracy(latent, image):
    latent, image = np.asarray(latent, np.float64), np.asarray(image, np.float64)
    latent = latent - latent.mean(1, keepdims=True)
    image = image - image.mean(1, keepdims=True)
    latent /= np.maximum(np.linalg.norm(latent, axis=-1, keepdims=True), 1e-12)
    image /= np.maximum(np.linalg.norm(image, axis=-1, keepdims=True), 1e-12)
    sims = np.einsum("qid,qjd->qij", latent, image)
    return float((sims.argmax(-1) == np.arange(latent.shape[1])[None]).mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_stats.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/latent_image_stats.py tests/evaluation/test_latent_image_stats.py
git commit -m "feat: add clustered statistics for latent image analysis"
```

---

### Task 7: 보조 자유 생성 — E3 mask 이식

**Files:**
- Create: `src/evaluation/visual_interventions.py`
- Modify: `src/evaluation/continuous_latent.py` (`ContinuousLatentController.generate` 시그니처와 decoder 호출 두 곳, rollout 등록 한 곳)
- Modify: `src/stage2/inference.py` (`Stage2InferenceController.generate`에 `intervention=None` 전달)
- Test: `tests/evaluation/test_visual_interventions.py`, 기존 `tests/evaluation/test_continuous_latent.py`

**Interfaces:**
- Produces:
  - `build_image_attention_mask(validity_mask, image_positions, latent_query_positions, mode, dtype) -> torch.Tensor` (`mode` ∈ `{"causal_mask", "latent_mask"}`; 그 외 `ValueError`)
  - `class VisualIntervention(mode: str, image_token_id: int)` with `begin(input_ids, token_ids)`, `register_latent_span(span)`, `wrap_decoder(language_model)`, `forward_records: list[dict]`
  - `ContinuousLatentController.generate(..., intervention: VisualIntervention | None = None)`; `None`이면 기존 동작과 바이트 단위로 같은 호출.

이식 원본: `/root/workspace/ILVR/.worktrees/ours1-visual-causal/src/evaluation/visual_interventions.py` (`build_image_attention_mask`, `VisualIntervention`의 `begin/register_latent_span/wrap_decoder/_forward_role`만). `global_mask`, `suppress_start`, `process_logits`는 가져오지 않는다. sliding window 인자는 제거한다(config `use_sliding_window=false`).

`continuous_latent.py` 변경 (이 branch 기준 `generate` 본문):

```python
# signature: add after disable_latent_generation
        intervention=None,
# after `inputs_embeds = model.prepare_stage2_base_inputs_embeds(...)`
        decoder = partial(model.model, generate_mode=True)
        if intervention is not None:
            intervention.begin(input_ids, token_ids)
            decoder = intervention.wrap_decoder(decoder)
# text-step forward: replace `outputs = model.model(... generate_mode=True)` with
            outputs = decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
# inside the START branch, before rollout
                if intervention is not None:
                    intervention.register_latent_span(span)
# rollout call: language_model=decoder
```

`return_post_end_logits` 경로의 `model.model(...)` 호출도 `decoder(...)`로 바꾼다 (generate_mode는 partial에 포함).

- [ ] **Step 1: Write the failing test**

```python
# tests/evaluation/test_visual_interventions.py
import pytest
import torch
from evaluation.visual_interventions import VisualIntervention, build_image_attention_mask


def test_latent_mask_blocks_only_latent_rows():
    m = build_image_attention_mask(torch.ones(1, 8), (1, 2), (5, 6), "latent_mask", torch.float32)
    allowed = m[0, 0] == 0
    assert not allowed[5, 1] and not allowed[6, 2] and allowed[4, 1] and allowed[7, 2]
    assert torch.equal(allowed, allowed.tril() & allowed)


def test_causal_control_is_plain_causal():
    m = build_image_attention_mask(torch.ones(1, 5), (1,), (3,), "causal_mask", torch.float32)
    assert torch.equal(m[0, 0] == 0, torch.ones(5, 5, dtype=torch.bool).tril())


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="unknown"):
        build_image_attention_mask(torch.ones(1, 3), (), (), "global_mask", torch.float32)


def test_wrapper_injects_4d_mask():
    calls = []
    iv = VisualIntervention("latent_mask", image_token_id=80)
    iv.begin(torch.tensor([[1, 80, 2]]), None)
    wrapped = iv.wrap_decoder(lambda **kw: calls.append(kw) or "ok")
    assert wrapped(inputs_embeds=torch.zeros(1, 3, 4), attention_mask=torch.ones(1, 3),
                   position_ids=None, use_cache=False, return_dict=True) == "ok"
    assert calls[0]["attention_mask"].shape == (1, 1, 3, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_visual_interventions.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Port the module**

```python
# src/evaluation/visual_interventions.py
"""Generation-local image masks for latent-only image access tests.

Ported subset of ours1-visual-causal src/evaluation/visual_interventions.py
(E3 latent mask and its same-format causal control only).
"""
from __future__ import annotations

from functools import wraps

import torch

_MODES = ("causal_mask", "latent_mask")


def _mode(mode):
    if mode not in _MODES:
        raise ValueError(f"unknown visual intervention mode {mode!r}; expected one of {', '.join(_MODES)}")
    return mode


def identify_image_positions(input_ids, image_token_id):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise NotImplementedError("visual interventions currently support batch_size=1 only")
    return tuple((input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten().tolist())


def build_image_attention_mask(validity_mask, image_positions, latent_query_positions, mode, dtype):
    """Finite additive causal mask; latent_mask removes image keys from latent query rows."""
    mode = _mode(mode)
    if validity_mask.ndim != 2 or validity_mask.shape[0] != 1:
        raise NotImplementedError("visual attention masks currently support batch_size=1 only")
    if not dtype.is_floating_point:
        raise TypeError("attention mask dtype must be floating point")
    length, device = validity_mask.shape[1], validity_mask.device
    query = torch.arange(length, device=device)[:, None]
    key = torch.arange(length, device=device)[None, :]
    allowed = (key <= query) & validity_mask[0].to(torch.bool)[None, :]
    images = tuple(p for p in image_positions if 0 <= p < length)
    queries = tuple(p for p in latent_query_positions if 0 <= p < length)
    if mode == "latent_mask" and images and queries:
        allowed[torch.tensor(queries, device=device)[:, None], torch.tensor(images, device=device)[None, :]] = False
    additive = torch.full((length, length), torch.finfo(dtype).min, dtype=dtype, device=device)
    additive.masked_fill_(allowed, 0)
    return additive[None, None]


class VisualIntervention:
    """Apply one image mask mode to every no-cache decoder call of one generation."""

    def __init__(self, mode, image_token_id):
        self.mode = _mode(mode)
        self.image_token_id = image_token_id
        self.forward_records = []
        self._image_positions = ()
        self._latent_query_positions = []

    def begin(self, input_ids, token_ids):
        self.forward_records = []
        self._latent_query_positions = []
        self._image_positions = identify_image_positions(input_ids, self.image_token_id)

    def register_latent_span(self, span):
        for position in (span.start_position, *span.latent_positions):
            if position not in self._latent_query_positions:
                self._latent_query_positions.append(position)

    def wrap_decoder(self, language_model):
        @wraps(language_model)
        def wrapped(**kwargs):
            if kwargs.get("past_key_values") is not None or kwargs.get("use_cache") is not False:
                raise ValueError("visual intervention decoder calls require use_cache=False and no cache")
            embeds, mask = kwargs.get("inputs_embeds"), kwargs.get("attention_mask")
            if embeds is None or embeds.ndim != 3 or mask is None or mask.ndim != 2 or mask.shape != embeds.shape[:2]:
                raise ValueError("expected [batch, sequence, hidden] embeddings with a matching 2-D validity mask")
            prefix = embeds.shape[1]
            self.forward_records.append({
                "prefix_length": prefix,
                "role": "latent_rollout" if prefix - 1 in self._latent_query_positions else "text",
                "latent_query_positions": tuple(p for p in self._latent_query_positions if p < prefix),
            })
            kwargs["attention_mask"] = build_image_attention_mask(
                mask, self._image_positions, tuple(self._latent_query_positions), self.mode, embeds.dtype)
            return language_model(**kwargs)
        return wrapped
```

- [ ] **Step 4: Wire `intervention` through `continuous_latent.py` and `stage2/inference.py`** exactly as shown above.

- [ ] **Step 5: Run new and existing tests**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_visual_interventions.py tests/evaluation/test_continuous_latent.py -v`
Expected: all pass (existing continuous-latent tests unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/visual_interventions.py src/evaluation/continuous_latent.py src/stage2/inference.py tests/evaluation/test_visual_interventions.py
git commit -m "feat: allow latent-only image masks during continuous generation"
```

---

### Task 8: CLI 실행 단계와 게이트

**Files:**
- Create: `src/evaluation/latent_image_cli.py`, `scripts/analyze_latent_image.py`
- Test: `tests/evaluation/test_latent_image_cli.py`

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces: 명령 `scripts/analyze_latent_image.py <stage> --run-dir results/latent-image-attention-<ts> [--shard i --num-shards n]`.

공통 인자: `--model-dir`, `--base-model-dir`, `--data data/TEST.jsonl`, `--selection <metadata.json>`, `--eval-rows <merged.jsonl>`, `--data-root data`, `--run-dir`, `--shard`, `--num-shards`. `prepare`가 입력 SHA256과 `git rev-parse HEAD`를 `run-dir/manifest.json`에 기록하고, 이후 단계는 manifest의 해시·commit과 현재 값이 다르면 중단한다.

| stage | 입력 | 출력 | 동작 |
|---|---|---|---|
| `prepare` | 공통 인자 | `manifest.json`, `swaps.json` | 해시 검증 (Global Constraints 값과 불일치 시 중단), category pool, 문항별 swap 4개 선택 |
| `smoke` | manifest | `smoke/gates.json` | 문항 2개 (block 1개인 첫 문항, block ≥3인 첫 문항) 에서 G1–G5 |
| `attention` | manifest, smoke 통과 | `attention/q{idx}.npz`, `attention/q{idx}.json` | G5 reference → 제외 판정 → ③ capture |
| `heads` | 전 shard attention | `heads.json` | H = latent pad 행 value-weighted image 비율 평균 상위 29 (layer, head) |
| `interventions` | manifest, heads.json | `interventions/q{idx}.json`, `interventions/q{idx}.npz` | 아래 조건 목록 |
| `freegen` | manifest | `freegen/{mode}/q{idx}.json` | `causal_mask`, `latent_mask` 두 모드 greedy 생성 + `score_answer` |
| `report` | 모든 출력 | `summary.json`, `report.md`, `figures/` | Task 9 |

게이트 (spec §6, `smoke`에서 판정, 본 단계에서도 문항마다 기록):
- G1: `capture.g1_max_rel ≤ 1e-2`.
- G2: `capture.g2_max_abs ≤ 1e-3`.
- G3: `noop` vs `reference(native)` — scored 위치 argmax 불일치 0, 최대 |Δnll| ≤ 1e-2, latent 최대 상대차 ≤ 1e-2.
- G4: `dual_clean` (blur alt) vs `noop` — G3와 같은 기준.
- G5: `reference`의 scored 위치 argmax와 저장 token 일치율. 첫 block 이전 불일치 또는 일치율 < 0.99 → 문항 제외 (`excluded.json`에 사유). 제외 > 10 → `attention`/`interventions` 단계 중단, 비정상 종료.

`interventions` 조건 목록 (문항마다, 모두 `numerical_policy()` 안):

```python
conditions = [
    Condition("reference", native=True),
    Condition("noop"), Condition("noop_repeat"),
    Condition("k_all", latent_source="none"),
    Condition("k_heads", head_subset=heads),
    Condition("k_random", random_keys=True),
]
for kind in ("blur", "mean", "swap"):
    conditions += [Condition(f"{kind}_dual_clean"), Condition(f"{kind}_total", "alt", "alt"),
                   Condition(f"{kind}_direct", "clean", "alt"), Condition(f"{kind}_indirect", "alt", "clean")]
# B: pixel_override runs for swaps[0..3] -> names "cand1".."cand4"
```

`interventions/q{idx}.json` 내용 (noop 기준; block별 리스트):
- `latent_cos[cond][b]` = pad별 `cosine_distance(latents)` 평균, `latent_cos_pads[cond][b]` (8개)
- `post_dnll[cond][b]`, `post_flip[cond][b]` (`post_span[b]` 평균; 비면 `null`)
- `answer_dnll[cond]`, `answer_flip[cond]` (`answer_status != "ok"`이면 `null` + 사유)
- `control_cos[cond]` = pre/post control 위치 최종 hidden cosine 거리 평균 (B 민감도의 text 대조)
- `random_matched[b]`, `excluded_reason`, gate 수치
- npz: `mid_body`, `final_body`, `image_mid`, `image_final` for `noop`, `cand1..4` (B 식별; 후보 순서 = [원본, cand1..cand4]).

`freegen`: `eval.load_evaluation_artifacts`로 로드(TF32 on, 기존 eval과 같은 수치). 생성 설정 = `build_stage2_generation_config(..., generation_kwargs={"max_new_tokens": 4096, "temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0, "max_sequence_length": 128000, "generation_limit_policy": "external_full_latent_block"})`. `eval.run_one_example`과 같은 순서로 prompt를 만들고 `Stage2InferenceController().generate(..., intervention=VisualIntervention(mode, image_token_id))`, `extract_assistant_content` → `score_answer(assistant_only, gold)`. 기록: `match`, `extracted_final_answer`, `latent_block_count`, `stop_reason`, `generated_suffix_token_ids`.

- [ ] **Step 1: Write the failing CPU tests** (model 없이 단계 로직만)

```python
# tests/evaluation/test_latent_image_cli.py
import json
import pytest
from evaluation import latent_image_cli as cli


def test_shard_assignment_is_disjoint_and_complete():
    items = tuple(range(10))
    shards = [cli.shard_items(items, i, 3) for i in range(3)]
    assert sorted(x for s in shards for x in s) == list(items)


def test_manifest_mismatch_aborts(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"inputs": {"data": "aaa"}, "commit": "c1"}))
    with pytest.raises(ValueError, match="manifest"):
        cli.require_manifest(tmp_path, {"data": "bbb"}, "c1")
    with pytest.raises(ValueError, match="manifest"):
        cli.require_manifest(tmp_path, {"data": "aaa"}, "c2")
    cli.require_manifest(tmp_path, {"data": "aaa"}, "c1")


def test_g5_exclusion_rules():
    ok = cli.g5_verdict(matches=[True] * 200, first_block_offset=5)
    assert ok["excluded"] is False
    early = cli.g5_verdict(matches=[True, False] + [True] * 198, first_block_offset=5)
    assert early["excluded"] and early["reason"] == "mismatch_before_first_block"
    low = cli.g5_verdict(matches=[True] * 10 + [False] * 5 + [True] * 85, first_block_offset=5)
    assert low["excluded"] and low["reason"] == "match_rate_below_0.99"


def test_too_many_exclusions_abort():
    with pytest.raises(RuntimeError, match="exclusions"):
        cli.require_exclusion_budget(11, limit=10)
    cli.require_exclusion_budget(10, limit=10)


def test_head_selection_top_fraction():
    import numpy as np
    share = np.zeros((36, 16)); share[3, 4] = 1; share[10, 2] = .5
    heads = cli.select_heads(share, fraction=0.05)
    assert len(heads) == 29 and heads[0] == (3, 4) and heads[1] == (10, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_cli.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the helpers**

```python
# src/evaluation/latent_image_cli.py  (helpers; stage functions follow in Step 5)
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np


def shard_items(items, shard, num_shards):
    return tuple(x for i, x in enumerate(items) if i % num_shards == shard)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def require_manifest(run_dir, inputs, commit):
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    if manifest["inputs"] != inputs or manifest["commit"] != commit:
        raise ValueError("manifest inputs or commit differ from the current execution")
    return manifest


def g5_verdict(matches, first_block_offset):
    rate = sum(matches) / len(matches)
    first_bad = next((i for i, ok in enumerate(matches) if not ok), None)
    if first_bad is not None and first_bad < first_block_offset:
        return {"excluded": True, "reason": "mismatch_before_first_block", "rate": rate, "first_mismatch": first_bad}
    if rate < 0.99:
        return {"excluded": True, "reason": "match_rate_below_0.99", "rate": rate, "first_mismatch": first_bad}
    return {"excluded": False, "reason": None, "rate": rate, "first_mismatch": first_bad}


def require_exclusion_budget(count, limit=10):
    if count > limit:
        raise RuntimeError(f"{count} trajectory exclusions exceed the limit of {limit}")


def select_heads(share, fraction=0.05):
    count = int(round(share.size * fraction))          # 576 * 0.05 = 28.8 -> 29
    order = np.argsort(-share, axis=None, kind="stable")[:count]
    return [tuple(int(v) for v in np.unravel_index(i, share.shape)) for i in order]
```

G5 `first_block_offset` is the index of the first latent START within `layout.scored_positions`.

- [ ] **Step 4: Run helper tests**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_cli.py -v`
Expected: 5 passed

- [ ] **Step 5: Write failing tests for the per-question logic** (fake results, no model)

```python
# append to tests/evaluation/test_latent_image_cli.py
import torch
from evaluation.latent_image_interventions import ConditionResult
from evaluation.latent_image_layout import LayoutTokenIds, build_layout

T = LayoutTokenIds(80, 81, 83, 84, 91, 90, 92, (70, 71))
PROMPT = [1, 2, 81, 80, 80, 83, 5, 6, 84, 7, 8]


def fake(name, shift=0.0, flip_at=None):
    lay = build_layout(PROMPT + [10, 91, 90, 90, 92, 13, 14, 70, 71, 5, 84], len(PROMPT), T, latent_size=2)
    scores = {p: ((1 if p == flip_at else 0), 1.0 + shift) for p in lay.scored_positions}
    lat = torch.ones(1, 2, 4) + shift * torch.tensor([1.0, -1, 0, 0])
    hidden = {p: torch.ones(4) + shift for p in lay.pre_text[0] + lay.post_text[0]}
    return lay, ConditionResult(name, lat, torch.ones(1, 4), torch.ones(1, 4), torch.ones(4), torch.ones(4),
                                hidden, scores)


def test_compare_runs_gate():
    _, a = fake("noop")
    _, b = fake("reference")
    assert cli.compare_runs(a, b)["pass"]
    _, c = fake("reference", flip_at=16)
    assert not cli.compare_runs(c, a)["pass"]


def test_summarize_question_deltas():
    lay, base = fake("noop")
    _, other = fake("k_all", shift=0.5, flip_at=16)
    out = cli.summarize_question({"noop": base, "k_all": other}, lay, {0: 1.0})
    assert out["post_dnll"]["k_all"] == [0.5] and out["post_flip"]["k_all"][0] == pytest.approx(0.5)
    assert out["answer_dnll"]["k_all"] == pytest.approx(0.5)
    assert out["latent_cos"]["k_all"][0] > 0 and "noop" not in out["latent_cos"]
```

(`post_span[0]` = positions 16, 17; flip at 16 → flip rate 0.5.)

- [ ] **Step 6: Implement the per-question functions**

```python
# append to src/evaluation/latent_image_cli.py
from evaluation.latent_image_stats import cosine_distance


def compare_runs(a, b, tol=1e-2):
    tokens = sorted(b.scores)
    mismatches = sum(a.scores[p][0] != b.scores[p][0] for p in tokens)
    max_dnll = max(abs(a.scores[p][1] - b.scores[p][1]) for p in tokens)
    rel = float(((a.latents - b.latents).norm(dim=-1) / b.latents.norm(dim=-1).clamp_min(1e-6)).max())
    return {"argmax_mismatches": int(mismatches), "max_abs_dnll": float(max_dnll), "max_latent_rel": rel,
            "pass": mismatches == 0 and max_dnll <= tol and rel <= tol}


def g5_for(reference, trajectory):
    ids = trajectory.input_ids[0].tolist()
    scored = trajectory.layout.scored_positions
    matches = [reference.scores[p][0] == ids[p] for p in scored]
    return g5_verdict(matches, scored.index(trajectory.layout.blocks[0].start))


def summarize_question(results, layout, matched):
    base = results["noop"]
    out = {key: {} for key in ("latent_cos", "latent_cos_pads", "post_dnll", "post_flip",
                               "answer_dnll", "answer_flip", "control_cos")}
    out.update(random_matched=matched, answer_status=layout.answer_status)
    controls = sorted(base.last_hidden)

    def delta(result, positions):
        if not positions:
            return None, None
        d = [result.scores[p][1] - base.scores[p][1] for p in positions]
        f = [result.scores[p][0] != base.scores[p][0] for p in positions]
        return float(np.mean(d)), float(np.mean(f))

    for name, result in results.items():
        if name == "noop":
            continue
        pads = cosine_distance(result.latents.numpy(), base.latents.numpy())
        out["latent_cos_pads"][name] = pads.tolist()
        out["latent_cos"][name] = pads.mean(1).tolist()
        post = [delta(result, span) for span in layout.post_span]
        out["post_dnll"][name] = [p[0] for p in post]
        out["post_flip"][name] = [p[1] for p in post]
        answer = delta(result, layout.answer_positions) if layout.answer_status == "ok" else (None, None)
        out["answer_dnll"][name], out["answer_flip"][name] = answer
        out["control_cos"][name] = (float(np.mean([cosine_distance(result.last_hidden[p].numpy(),
                                                                   base.last_hidden[p].numpy()) for p in controls]))
                                    if controls else None)
    return out


def question_interventions(model, processor, trajectory, heads, swap_images):
    from evaluation.latent_image_corruptions import (blur_image, mean_image, pixel_inputs,
                                                     require_same_grid, resized_swap)
    from evaluation.latent_image_interventions import Condition, random_key_sets, run_condition
    original = trajectory.images[0]

    def pix(image, label):
        values, grid = pixel_inputs(processor, image)
        require_same_grid(trajectory.image_grid_thw, grid, label)
        return values, grid

    ip = processor.image_processor
    alts = {"blur": pix(blur_image(original), "blur"), "mean": pix(mean_image(original), "mean"),
            "swap": pix(resized_swap(swap_images[0], original, ip), "swap")}
    candidates = [pix(resized_swap(image, original, ip), f"cand{i + 1}") for i, image in enumerate(swap_images)]
    random_sets, matched = random_key_sets(trajectory.layout, source_index=trajectory.source_index)
    results = {}

    def run(condition, **kwargs):
        results[condition.name] = run_condition(model, trajectory, condition, **kwargs)

    run(Condition("reference", native=True))
    run(Condition("noop"))
    run(Condition("noop_repeat"))
    run(Condition("k_all", latent_source="none"))
    run(Condition("k_heads", head_subset=tuple(tuple(h) for h in heads)))
    run(Condition("k_random", random_keys=True), random_blocked=random_sets)
    for kind, (values, grid) in alts.items():
        for name, text, latent in (("dual_clean", "clean", "clean"), ("total", "alt", "alt"),
                                   ("direct", "clean", "alt"), ("indirect", "alt", "clean")):
            run(Condition(f"{kind}_{name}", text, latent), alt_pixel_values=values, alt_grid=grid)
    for i, (values, grid) in enumerate(candidates):
        run(Condition(f"cand{i + 1}"), pixel_override=(values, grid))
    gates = {"G3": compare_runs(results["noop"], results["reference"]),
             "G4": compare_runs(results["blur_dual_clean"], results["noop"]),
             "G5": g5_for(results["reference"], trajectory)}
    return results, matched, gates


def content_arrays(results):
    names = ["noop", "cand1", "cand2", "cand3", "cand4"]
    return {key: np.stack([getattr(results[n], key).numpy() for n in names]).astype(np.float32)
            for key in ("mid_body", "final_body", "image_mid", "image_final")}


def heatmap_questions(selection, categories):
    chosen = [next(i for i in selection if categories[i] == c) for c in ("creation", "deletion", "selection", "update")]
    return tuple(chosen + [i for i in selection if i not in chosen][:2])


def question_freegen(model, processor, test_row, data_root, mode):
    from eval import extract_assistant_content
    from evaluation.latent_image_trajectory import _open_images
    from evaluation.runtime import prepare_evaluation_input
    from evaluation.scoring import score_answer
    from evaluation.visual_interventions import VisualIntervention
    from stage2.inference import Stage2InferenceController, build_stage2_generation_config
    from stage2.peft import resolve_stage2_qwen_model
    from stage2.checkpointing import resolve_latent_token_ids  # same helper eval.py uses
    qwen = resolve_stage2_qwen_model(model)
    images, _ = _open_images(test_row, data_root)
    prepared = prepare_evaluation_input(processor, test_row, images)
    device = qwen.model.embed_tokens.weight.device
    inputs = {k: v.to(device) for k, v in prepared.tensors.items()}
    config = build_stage2_generation_config(
        model_config=qwen.config, model_generation_config=getattr(qwen, "generation_config", None),
        tokenizer=processor.tokenizer,
        generation_kwargs={"max_new_tokens": 4096, "temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0,
                           "max_sequence_length": 128000, "generation_limit_policy": "external_full_latent_block"})
    with torch.inference_mode():
        result = Stage2InferenceController().generate(
            model=qwen, input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            token_ids=resolve_latent_token_ids(processor.tokenizer, qwen.config), generation_config=config,
            pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"],
            intervention=VisualIntervention(mode, qwen.config.image_token_id))
    suffix = result.sequences[:, prepared.prompt_token_length:]
    assistant = extract_assistant_content(processor.batch_decode(suffix, skip_special_tokens=False)[0].strip())
    score = score_answer(assistant, test_row.get("original_final_answer"))
    return {"match": bool(score.match), "extracted_final_answer": score.prediction.extracted,
            "latent_block_count": int(result.latent_block_count), "stop_reason": result.stop_reason,
            "generated_suffix_token_ids": suffix[0].tolist(), "prompt_identity": prepared.prompt_identity}
```

Add `import torch` at the top of the module. Import paths verified at plan time: `resolve_latent_token_ids` in `src/stage2/checkpointing.py:1245`, `extract_assistant_content` in `eval.py:303`, `AnswerScore.prediction.extracted` in `src/evaluation/scoring.py:40-53`.

- [ ] **Step 7: Run the CLI tests**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_cli.py -v`
Expected: 7 passed

- [ ] **Step 8: Implement `main` and the stage drivers**

Implement `main(argv=None)` with `argparse` subcommands matching the table. Each stage:
1. `prepare`: compute `sha256` of data/selection/eval rows, compare to the Global Constraints values, write `manifest.json` `{"inputs": {...}, "commit": git_commit(), "model_dir", "base_model_dir", "prepared_manifest_sha256": sha256(<model_dir>.manifest.json), "adapter_sha256": sha256(<model_dir>/adapter_model.safetensors), "created": <UTC ISO>}`, then load TEST rows, build category pool over the 100 selected indices (from `image_input` path), `select_swaps` for each, write `swaps.json`.
2. Teacher-forced stages (`smoke`, `attention`, `interventions`) load once via `evaluation.vision_block_analysis._load_runtime(args)` (exactly one visible GPU; SDPA + FA2 vision; BF16; `args` needs `model_dir`, `base_model_dir`) and wrap every question in `with numerical_policy():`. Per owned question: `build_trajectory(processor, test_rows[i], eval_rows[i], i, Path("data"), layout_token_ids(processor, qwen.config))`.
3. `smoke`: questions = first selected index with exactly 1 block and first with ≥3 blocks (in selection order). Run `capture_attention` (G1 = `g1_max_rel ≤ 1e-2`, G2 = `g2_max_abs ≤ 1e-3`) and `question_interventions` (G3, G4, G5). Write `smoke/gates.json` `{source_index: {G1..G5}}` and `"pass": all(...)`; exit code 1 if any gate fails.
4. `attention`: per question run `Condition("reference", native=True)` → `g5_for`; excluded questions go to `attention/excluded/q{idx}.json`. Otherwise `capture_attention(keep_image_probs=idx in heatmap_questions(...))`, write `attention/q{idx}.npz` (`mass`, `vw_share`, `enrichment`, optional `image_probs`, `image_grid_thw`) and `attention/q{idx}.json` (`specs` as dicts, `g1`, `g2`, `g5`, `category`, `n_blocks`, `eval_match`). After the shard finishes, count all `excluded/*.json` and call `require_exclusion_budget`.
5. `heads`: over all `attention/q*.npz`, take rows whose spec slot starts with `pad`, average `vw_share[..., IMAGE]` over rows and questions → `[layers, heads]`; `select_heads(share)`; write `heads.json` `{"heads": [[layer, head], ...], "share": share.tolist()}`.
6. `interventions`: requires `heads.json`; skip excluded questions; swap images = `open_images` of the `swaps.json` entries for the question (first image of each TEST row); `question_interventions` → `{**summarize_question(...), "gates": gates}` into `interventions/q{idx}.json`, `content_arrays` into `interventions/q{idx}.npz`.
7. `freegen`: loads with `eval.load_evaluation_artifacts(model_dir=..., base_model_dir=..., cache_dir="data/cache")` (eval numerics), for `mode in ("causal_mask", "latent_mask")` writes `freegen/{mode}/q{idx}.json` from `question_freegen`.
8. Write per-question files atomically (`.tmp` then `os.replace`); skip questions whose output already exists (resume). Every stage except `prepare` first calls `require_manifest`.
9. `scripts/analyze_latent_image.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.latent_image_cli import main
raise SystemExit(main())
```

- [ ] **Step 9: Dry check the entry point**

Run: `./venv/ilvr-venv/bin/python scripts/analyze_latent_image.py --help`
Expected: lists the seven stages.

- [ ] **Step 10: Commit**

```bash
git add src/evaluation/latent_image_cli.py scripts/analyze_latent_image.py tests/evaluation/test_latent_image_cli.py
git commit -m "feat: add staged CLI and gates for latent image analysis"
```

---

### Task 9: 보고서 생성

**Files:**
- Create: `src/evaluation/latent_image_report.py`
- Test: `tests/evaluation/test_latent_image_report.py`

**Interfaces:**
- Consumes: Task 6 statistics, Task 8 output files.
- Produces: `build_report(run_dir: Path) -> dict` (writes `summary.json`, `report.md`, `figures/*.png`).

내용:
1. **③ 기술적 증거**: layer별 value-weighted image 비율 — latent(pad 평균), START, END, pre_text, post_text 곡선과 95% cluster CI. layer별 paired 차이 (pad − pre, pad − post). 판정: 두 차이의 CI 하한이 모두 0보다 큰 layer 목록. mass·enrichment 보조 표. head 상위 29 표.
2. **① knockout**: `k_all`, `k_heads`, `k_random`의 latent_cos, post_dnll, post_flip, answer_dnll (첫 block만 / 전체 block). `k_all − k_random` paired 차이.
3. **② patching**: corruption별 total/direct/indirect 표와 direct 비율 (`ratio_cluster_bootstrap(direct, total)`), latent_cos와 answer_dnll 기준.
4. **B**: latent_cos(total) vs control_cos(total) paired 차이; `identification_accuracy` (layer 18, final) vs 0.2, 문항 cluster bootstrap.
5. **자유 생성**: causal_mask 대조 vs latent_mask 정답 수, exact McNemar; eval 원본 정답(49/100)도 병기.
6. **게이트/제외**: G1–G5 수치, 제외 문항과 사유, random matched 비율 분포.
7. 각 절 제목에 증거 수준(기술적/인과/정보 내용/정답률)을 명시하고, 실행되지 않았거나 게이트 실패한 항목은 `NOT RUN` / `GATE FAILED`로 표기.

- [ ] **Step 1: Write the failing test** — synthetic run dir with 3 questions × 2 blocks of fabricated npz/json matching Task 8 schemas; assert `summary.json` keys `{"attention", "knockout", "patching", "content", "freegen", "gates"}`, that a missing `freegen/` yields `"freegen": {"status": "NOT RUN"}`, and that `report.md` contains `NOT RUN`.

```python
# tests/evaluation/test_latent_image_report.py
import json
import numpy as np
from evaluation.latent_image_report import build_report
from tests.evaluation.latent_image_fixtures import write_synthetic_run


def test_report_sections_and_not_run(tmp_path):
    write_synthetic_run(tmp_path, questions=3, blocks=2, with_freegen=False)
    summary = build_report(tmp_path)
    assert set(summary) >= {"attention", "knockout", "patching", "content", "freegen", "gates"}
    assert summary["freegen"] == {"status": "NOT RUN"}
    assert "NOT RUN" in (tmp_path / "report.md").read_text()
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
```

Create the fixture writer first; it fixes the on-disk schema shared with Task 8.

```python
# tests/evaluation/latent_image_fixtures.py
"""Synthetic latent-image run directories with the Task 8 on-disk schema."""
import json
from pathlib import Path

import numpy as np

LAYERS, HEADS, HIDDEN = 4, 2, 6
KNOCKOUT = ("reference", "noop_repeat", "k_all", "k_heads", "k_random")
PATCH = tuple(f"{k}_{n}" for k in ("blur", "mean", "swap") for n in ("dual_clean", "total", "direct", "indirect"))
CANDS = ("cand1", "cand2", "cand3", "cand4")


def write_synthetic_run(run_dir, questions=3, blocks=2, with_freegen=False, image_boost=0.3):
    run_dir, rng = Path(run_dir), np.random.default_rng(0)
    (run_dir / "manifest.json").write_text(json.dumps({"inputs": {}, "commit": "c"}))
    (run_dir / "heads.json").write_text(json.dumps({"heads": [[0, 1]], "share": np.zeros((LAYERS, HEADS)).tolist()}))
    for sub in ("attention", "interventions"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    for q in range(questions):
        specs = []
        for b in range(blocks):
            specs += [{"position": 0, "group": "latent", "block": b, "slot": s}
                      for s in ("start", "pad1", "pad2", "end")]
            specs += [{"position": 0, "group": g, "block": b, "slot": s}
                      for g, s in (("pre_text", "pre"), ("post_text", "post"))]
        rows = len(specs)
        share = rng.dirichlet(np.ones(7), size=(rows, LAYERS, HEADS)).astype(np.float16)
        for r, spec in enumerate(specs):          # latent pads look more at the image (region 2)
            if spec["slot"].startswith("pad"):
                share[r, :, :, 2] += np.float16(image_boost)
        np.savez(run_dir / "attention" / f"q{q}.npz", mass=share, vw_share=share,
                 enrichment=rng.random((rows, LAYERS, HEADS)).astype(np.float32),
                 image_grid_thw=np.array([[1, 2, 2]]))
        (run_dir / "attention" / f"q{q}.json").write_text(json.dumps(
            {"specs": specs, "g1": 1e-3, "g2": 1e-4, "g5": {"excluded": False, "rate": 1.0},
             "category": "deletion", "n_blocks": blocks, "eval_match": bool(q % 2)}))
        names = KNOCKOUT + PATCH + CANDS
        out = {key: {} for key in ("latent_cos", "latent_cos_pads", "post_dnll", "post_flip",
                                   "answer_dnll", "answer_flip", "control_cos")}
        for name in names:
            out["latent_cos"][name] = rng.random(blocks).tolist()
            out["latent_cos_pads"][name] = rng.random((blocks, 2)).tolist()
            out["post_dnll"][name] = rng.normal(size=blocks).tolist()
            out["post_flip"][name] = rng.random(blocks).tolist()
            out["answer_dnll"][name] = float(rng.normal())
            out["answer_flip"][name] = float(rng.random())
            out["control_cos"][name] = float(rng.random() * 0.1)
        out.update(random_matched={str(b): 1.0 for b in range(blocks)}, answer_status="ok",
                   gates={"G3": {"pass": True}, "G4": {"pass": True}, "G5": {"excluded": False, "rate": 1.0}})
        (run_dir / "interventions" / f"q{q}.json").write_text(json.dumps(out))
        image = rng.normal(size=(5, HIDDEN))
        np.savez(run_dir / "interventions" / f"q{q}.npz",
                 mid_body=(image[:, None] + 0.01 * rng.normal(size=(5, blocks, HIDDEN))).astype(np.float32),
                 final_body=rng.normal(size=(5, blocks, HIDDEN)).astype(np.float32),
                 image_mid=image.astype(np.float32), image_final=rng.normal(size=(5, HIDDEN)).astype(np.float32))
    if with_freegen:
        for mode in ("causal_mask", "latent_mask"):
            (run_dir / "freegen" / mode).mkdir(parents=True, exist_ok=True)
            for q in range(questions):
                (run_dir / "freegen" / mode / f"q{q}.json").write_text(
                    json.dumps({"match": mode == "causal_mask" or q == 0}))
```

Add two more tests to `tests/evaluation/test_latent_image_report.py`:

```python
def test_attention_detects_image_preference_and_identification(tmp_path):
    write_synthetic_run(tmp_path, questions=12, blocks=2, image_boost=0.5)
    summary = build_report(tmp_path)
    assert summary["attention"]["significant_layers"] == [0, 1, 2, 3]
    assert summary["content"]["identification"]["mid"]["mean"] == 1.0


def test_freegen_mcnemar(tmp_path):
    write_synthetic_run(tmp_path, questions=4, with_freegen=True)
    fg = build_report(tmp_path)["freegen"]
    assert fg["correct"] == {"causal_mask": 4, "latent_mask": 1}
    assert fg["discordant"] == {"causal_only": 3, "latent_only": 0}
```

- [ ] **Step 2: Run to verify failure**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_report.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `build_report`**

```python
# src/evaluation/latent_image_report.py
"""Evidence-level summary for latent image analysis runs."""
import json
from pathlib import Path

import numpy as np

from evaluation import latent_image_stats as st
from evaluation.latent_image_layout import IMAGE

GROUPS = ("pad", "start", "end", "pre", "post")


def _files(run_dir, sub, suffix):
    return {int(p.stem[1:]): p for p in sorted((run_dir / sub).glob(f"q*.{suffix}"))}


def _units(npz, meta, field):
    """(question, block) units: per-group mean over rows of `field`[..., IMAGE] averaged over heads."""
    units = {g: {} for g in GROUPS}
    for q, path in npz.items():
        values = np.load(path)[field].astype(np.float32)[..., IMAGE].mean(-1)      # [rows, layers]
        for row, spec in enumerate(json.loads(meta[q].read_text())["specs"]):
            group = "pad" if spec["slot"].startswith("pad") else spec["slot"]
            units[group].setdefault((q, spec["block"]), []).append(values[row])
    return {g: {k: np.mean(v, 0) for k, v in d.items()} for g, d in units.items()}


def attention_section(run_dir):
    npz, meta = _files(run_dir, "attention", "npz"), _files(run_dir, "attention", "json")
    if not npz:
        return {"status": "NOT RUN"}
    units = _units(npz, meta, "vw_share")
    layers = next(iter(units["pad"].values())).shape[0]
    curves = {}
    for group, data in units.items():
        keys = sorted(data)
        if not keys:
            continue
        values, clusters = np.stack([data[k] for k in keys]), np.array([k[0] for k in keys])
        curves[group] = [st.cluster_bootstrap(values[:, l], clusters) for l in range(layers)]
    diffs, significant = {}, []
    for control in ("pre", "post"):
        keys = sorted(set(units["pad"]) & set(units[control]))
        a = np.stack([units["pad"][k] for k in keys])
        b = np.stack([units[control][k] for k in keys])
        clusters = np.array([k[0] for k in keys])
        diffs[f"pad_minus_{control}"] = [st.paired_cluster_bootstrap(a[:, l], b[:, l], clusters) for l in range(layers)]
    for l in range(layers):
        if all(diffs[f"pad_minus_{c}"][l]["lo"] > 0 for c in ("pre", "post")):
            significant.append(l)
    mass = _units(npz, meta, "mass")
    heads = json.loads((run_dir / "heads.json").read_text()) if (run_dir / "heads.json").exists() else None
    return {"status": "RUN", "evidence": "descriptive", "vw_image_share": curves, "differences": diffs,
            "significant_layers": significant,
            "mass_image_mean": {g: float(np.mean([v.mean() for v in d.values()])) for g, d in mass.items() if d},
            "heads": heads["heads"] if heads else None}


def _metric_units(docs, name, metric, first_block=False):
    values, clusters = [], []
    for q, doc in docs.items():
        v = doc[metric].get(name)
        if v is None:
            continue
        v = v[:1] if first_block and isinstance(v, list) else v
        for x in (v if isinstance(v, list) else [v]):
            if x is not None:
                values.append(x)
                clusters.append(q)
    return np.array(values, dtype=np.float64), np.array(clusters)


def _table(docs, names):
    table = {}
    for name in names:
        row = {}
        for metric in ("latent_cos", "post_dnll", "post_flip", "answer_dnll", "answer_flip"):
            v, c = _metric_units(docs, name, metric)
            row[metric] = st.cluster_bootstrap(v, c) if len(v) else None
        v, c = _metric_units(docs, name, "latent_cos", first_block=True)
        row["latent_cos_first_block"] = st.cluster_bootstrap(v, c) if len(v) else None
        table[name] = row
    return table


def _question_means(docs, name, metric):
    return {q: float(np.mean([x for x in (d[metric][name] if isinstance(d[metric][name], list) else [d[metric][name]])
                              if x is not None]))
            for q, d in docs.items() if d[metric].get(name) is not None}


def intervention_sections(run_dir):
    paths = _files(run_dir, "interventions", "json")
    if not paths:
        return ({"status": "NOT RUN"},) * 3
    docs = {q: json.loads(p.read_text()) for q, p in paths.items()}
    knockout = {"status": "RUN", "evidence": "causal", "table": _table(docs, ("noop_repeat", "k_all", "k_heads", "k_random"))}
    a, b = _question_means(docs, "k_all", "latent_cos"), _question_means(docs, "k_random", "latent_cos")
    qs = sorted(set(a) & set(b))
    knockout["k_all_minus_k_random_latent_cos"] = st.paired_cluster_bootstrap(
        [a[q] for q in qs], [b[q] for q in qs], np.array(qs))
    patching = {"status": "RUN", "evidence": "causal", "by_corruption": {}}
    for kind in ("blur", "mean", "swap"):
        names = tuple(f"{kind}_{n}" for n in ("total", "direct", "indirect"))
        entry = {"table": _table(docs, names)}
        for metric in ("latent_cos", "answer_dnll"):
            d, t = _question_means(docs, f"{kind}_direct", metric), _question_means(docs, f"{kind}_total", metric)
            qs = sorted(set(d) & set(t))
            entry[f"direct_ratio_{metric}"] = st.ratio_cluster_bootstrap(
                [d[q] for q in qs], [t[q] for q in qs], np.array(qs)) if qs else None
        patching["by_corruption"][kind] = entry
    content = {"status": "RUN", "evidence": "information content", "sensitivity": {}, "identification": {}}
    for kind in ("blur", "mean", "swap"):
        lat = _question_means(docs, f"{kind}_total", "latent_cos")
        ctl = {q: d["control_cos"][f"{kind}_total"] for q, d in docs.items()
               if d["control_cos"].get(f"{kind}_total") is not None}
        qs = sorted(set(lat) & set(ctl))
        content["sensitivity"][kind] = st.paired_cluster_bootstrap(
            [lat[q] for q in qs], [ctl[q] for q in qs], np.array(qs)) if qs else None
    arrays = {q: np.load(p) for q, p in _files(run_dir, "interventions", "npz").items()}
    for label, body, image in (("mid", "mid_body", "image_mid"), ("final", "final_body", "image_final")):
        per_q = [st.identification_accuracy(a[body].mean(1)[None], a[image][None]) for a in arrays.values()]
        content["identification"][label] = st.cluster_bootstrap(np.array(per_q), np.array(list(arrays)))
        content["identification"][label]["chance"] = 0.2
    return knockout, patching, content


def freegen_section(run_dir):
    modes = ("causal_mask", "latent_mask")
    if not all((run_dir / "freegen" / m).is_dir() for m in modes):
        return {"status": "NOT RUN"}
    results = {m: {int(p.stem[1:]): json.loads(p.read_text())["match"]
                   for p in (run_dir / "freegen" / m).glob("q*.json")} for m in modes}
    qs = sorted(set(results["causal_mask"]) & set(results["latent_mask"]))
    causal_only = sum(results["causal_mask"][q] and not results["latent_mask"][q] for q in qs)
    latent_only = sum(results["latent_mask"][q] and not results["causal_mask"][q] for q in qs)
    return {"status": "RUN", "evidence": "accuracy", "questions": len(qs),
            "correct": {m: int(sum(results[m][q] for q in qs)) for m in modes},
            "discordant": {"causal_only": causal_only, "latent_only": latent_only},
            "mcnemar_p": st.exact_mcnemar(causal_only, latent_only)}


def gates_section(run_dir):
    attention = {q: json.loads(p.read_text()) for q, p in _files(run_dir, "attention", "json").items()}
    interventions = {q: json.loads(p.read_text()).get("gates", {})
                     for q, p in _files(run_dir, "interventions", "json").items()}
    excluded = {int(p.stem[1:]): json.loads(p.read_text())
                for p in sorted((run_dir / "attention" / "excluded").glob("q*.json"))}
    smoke = run_dir / "smoke" / "gates.json"
    return {"smoke": json.loads(smoke.read_text()) if smoke.exists() else "NOT RUN",
            "g1_max": max((a["g1"] for a in attention.values()), default=None),
            "g2_max": max((a["g2"] for a in attention.values()), default=None),
            "g3_fail": [q for q, g in interventions.items() if not g.get("G3", {}).get("pass", False)],
            "g4_fail": [q for q, g in interventions.items() if not g.get("G4", {}).get("pass", False)],
            "excluded": excluded}


def _markdown(summary):
    lines = ["# Latent image attention analysis", ""]
    titles = {"attention": "③ Attention (descriptive evidence)", "knockout": "① Knockout (causal evidence)",
              "patching": "② Image patching (causal evidence)", "content": "B. Latent information content",
              "freegen": "Free generation (accuracy)", "gates": "Gates and exclusions"}
    for key, title in titles.items():
        section = summary[key]
        status = section.get("status", "RUN") if isinstance(section, dict) else "RUN"
        lines += [f"## {title}", "", f"Status: **{status}**", ""]
        if status != "NOT RUN":
            lines += ["```json", json.dumps(section, indent=1)[:20000], "```", ""]
    return "\n".join(lines)


def build_report(run_dir):
    run_dir = Path(run_dir)
    knockout, patching, content = intervention_sections(run_dir)
    summary = {"attention": attention_section(run_dir), "knockout": knockout, "patching": patching,
               "content": content, "freegen": freegen_section(run_dir), "gates": gates_section(run_dir)}
    if summary["gates"]["g3_fail"] or summary["gates"]["g4_fail"]:
        for key in ("knockout", "patching", "content"):
            if summary[key].get("status") == "RUN":
                summary[key]["status"] = "GATE FAILED"
    summary = json.loads(json.dumps(summary, default=float))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "report.md").write_text(_markdown(summary))
    return summary
```

The JSON dump in `report.md` is the raw evidence; Task 11 Step 6 turns it into the prose report for the user.

- [ ] **Step 4: Run to verify pass**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_report.py -v`
Expected: 3 passed

- [ ] **Step 5: Figures**

Load the `dataviz` skill first, then add `write_figures(run_dir, summary)` to `latent_image_report.py` and call it at the end of `build_report` (skip silently when a section is `NOT RUN`):
- `figures/attention_layers.png`: x = layer 0–35, y = value-weighted image share, one line per group (pad, start, end, pre, post) with CI bands from `summary["attention"]["vw_image_share"]`; mark `significant_layers`.
- `figures/patching.png`: per corruption, grouped bars total/direct/indirect of `latent_cos` mean with CI.
- `figures/heatmap_q{idx}.png`: for questions with `image_probs` in their npz, the layer-mean pad-row image attention reshaped to the merged grid `(h/2, w/2)` from `image_grid_thw`.
Test: extend `test_report_sections_and_not_run` with `assert (tmp_path / "figures" / "attention_layers.png").exists()`.

- [ ] **Step 6: Run to verify pass**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_latent_image_report.py -v`
Expected: 3 passed

- [ ] **Step 7: Full CPU suite for this feature**

Run: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation -q`
Expected: all pass (new and pre-existing).

- [ ] **Step 8: Commit**

```bash
git add src/evaluation/latent_image_report.py tests/evaluation/test_latent_image_report.py tests/evaluation/latent_image_fixtures.py
git commit -m "feat: summarize latent image analysis evidence levels"
```

---

### Task 10: GPU smoke

**Files:** 출력만 (`results/latent-image-attention-<ts>/`, `logs/latent-image-attention-<ts>/`).

- [ ] **Step 1: Prepare**

```bash
TS=$(date -u +%Y%m%d-%H%M%S); RUN=results/latent-image-attention-$TS; LOG=logs/latent-image-attention-$TS
mkdir -p $RUN $LOG
COMMON="--model-dir checkpoints/evaluation-staging/stage2-ce-matched-gb16-s212-20260919-155759/prepared-checkpoint-212 \
 --base-model-dir checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct --data data/TEST.jsonl \
 --selection /root/workspace/ILVR/results/server-comparison-20260912/analysis_completed/visual-causal-panel-20260910t115000z/ce/no_latent_execution/metadata.json \
 --eval-rows /root/workspace/ILVR/.worktrees/voff-reproduction-20260917/results/eval-stage2-ce-matched-test400-20260919-155759/merged.jsonl \
 --data-root data --run-dir $RUN"
./venv/ilvr-venv/bin/python scripts/analyze_latent_image.py prepare $COMMON 2>&1 | tee $LOG/prepare.log
```

Expected: `manifest.json`, `swaps.json` 생성, 해시 불일치 없음.

- [ ] **Step 2: Smoke on one GPU**

```bash
CUDA_VISIBLE_DEVICES=0 ./venv/ilvr-venv/bin/python scripts/analyze_latent_image.py smoke $COMMON 2>&1 | tee $LOG/smoke.log
```

Expected: `smoke/gates.json`에서 G1–G5 모두 pass. 하나라도 실패하면 **중단하고 사용자에게 수치와 함께 보고**한다 (기준 완화 금지).

- [ ] **Step 3: Record** — `smoke/gates.json` 요약을 사용자에게 보고한 뒤 Task 11로 진행.

---

### Task 11: 본실행과 보고

- [ ] **Step 1: ③ attention (2 GPU shard)**

```bash
for i in 0 1; do CUDA_VISIBLE_DEVICES=$i ./venv/ilvr-venv/bin/python scripts/analyze_latent_image.py attention $COMMON --shard $i --num-shards 2 > $LOG/attention-$i.log 2>&1 & done; wait
```

Expected: `attention/q*.json` 100개 − 제외 문항, `excluded.json` ≤ 10.

- [ ] **Step 2: Fix head set**

```bash
./venv/ilvr-venv/bin/python scripts/analyze_latent_image.py heads $COMMON 2>&1 | tee $LOG/heads.log
```

Expected: `heads.json` 29개.

- [ ] **Step 3: ①②B interventions (2 GPU shard)** — same loop with stage `interventions`, logs `interventions-$i.log`.

- [ ] **Step 4: Free generation (2 GPU shard)** — same loop with stage `freegen`, logs `freegen-$i.log`.

- [ ] **Step 5: Report**

```bash
./venv/ilvr-venv/bin/python scripts/analyze_latent_image.py report $COMMON 2>&1 | tee $LOG/report.log
```

Expected: `summary.json`, `report.md`, `figures/`. `results/`에 모델 artifact가 없는지 `find $RUN -name "*.safetensors" -o -name "*.bin" -o -name "*.pt"` 결과가 비었는지 확인.

- [ ] **Step 6: Report to the user** — 증거 수준별(기술적 ③ / 인과 ①② / 정보 내용 B / 정답률 자유 생성) 결과와 게이트·제외 현황을 따로 보고한다. 실행하지 않은 항목은 통과로 표시하지 않는다.
