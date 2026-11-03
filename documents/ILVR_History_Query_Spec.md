# ILVR 참조도·최근성 기반 Query 개선 — 구현 명세서

문서 버전: 1.0  
작성일: 2026-10-15  
기준 저장소: XD111ds/ILVR  
기준 커밋: `13a67ed2ee975d6ca56b4e9eba18d68b352b166b` (2026-08-05 커밋)  
권장 저장 위치: `docs/superpowers/specs/2026-10-15-ilvr-history-query.md`  
상태: 공개 소스 정적 검토를 바탕으로 작성한 구현 계약. 구현·학습·GPU 테스트를 실행한 결과가 아니다.

## 1. 결정 요약

**Stage 1의 supervision-target selector를 변경한다. 텍스트 구간에는 시간적 discount를, 과거 latent token에는 현재 reasoning의 attention 참조도를 적용한다.** Student의 추론 구조, candidate visual feature 생성, 기존 alignment loss는 변경하지 않는다.

구현의 핵심은 가중치 식보다 **attention을 측정할 teacher context를 올바르게 만드는 것**이다. 이전에 선택한 visual target을 실제 과거 latent 위치에 주입하고, 현재 latent 시작 직전의 prefix를 teacher에 통과시킨 뒤 참조도를 계산한다.

새 기능의 실행 모드는 `history_query_mode=adaptive`로 명시적으로 활성화한다. 기본값 `legacy`는 기존 selector를 유지한다. `gamma=0.8`, 마지막 decoder layer 사용은 첫 실험을 위한 설계 기본값이며 검증된 최적값이 아니다.

### 1.1. 논문 설명과 공개 구현을 구별할 것

논문 Eq. (4)는 누적 text/latent hidden-state 평균을 기술한다. 그러나 검사한 공개 구현은 이에 완전히 일치하지 않는다. [S0, S2]

| 항목 | 공개 구현에서 확인한 동작 | 본 명세의 변경 |
|---|---|---|
| Text query | 현재 `text_spans[seg_idx]`만 평균 | 구간별 평균에 discount를 적용하여 제한적으로 과거를 복원 |
| Latent history | `prev_sel_mean`: 직전 `chosen` visual feature의 평균 | 실제 주입된 모든 과거 latent 위치의 contextual hidden state를 재가중 |
| Teacher context | target 선택용 teacher 호출에 기존 `chosen`을 주입하지 않음 | 이전 target을 주입한 prefix를 순차적으로 재실행 |
| 선택 개수 | `_top_p_top_k`로 `Kstar <= K` | 가변 `Kstar`와 first-K mask 유지 |

따라서 저장소 기준으로는 **“누적 text 평균에 discount만 추가”하는 패치가 아니다.** 현재 구간 중심 구현에 discounted history를 도입하고, latent history의 표현·범위·teacher context를 함께 정비하는 변경이다. 이 차이를 실험 대조군에 반영한다.

## 2. 범위와 비범위

### 2.1. v1 지원 범위

- 저장소의 custom Qwen2.5-VL, Stage 1, full fine-tuning.
- 프로세스당 micro-batch size 1. 여러 프로세스의 data parallel / 기존 ZeRO-3 환경은 별도 통합 테스트를 통과해야 한다.
- 입력 이미지가 assistant 이전에 배치된 단일 assistant trajectory. 여러 입력 이미지와 여러 helper image를 지원한다.
- Teacher는 frozen/eval 상태의 rank-local EMA replica, student는 기존 attention backend를 유지한다.
- Attention은 마지막 layer, 마지막 4개 layer, 전체 decoder layer 중 하나에서 평균한다.
- 학습 target 생성에서만 작동하며, `eval.py`와 generation 로직에 추가 query 계산을 넣지 않는다.

### 2.2. 이번 변경에 포함하지 않는 것

Stage 2 학습 루프 재구현, Qwen3 지원, LoRA/quantization/FSDP/sequence parallel 지원, learnable fusion, attention rollout, gradient attribution, latent의 시간적 discount, KV-cache 최적화는 v1 범위 밖이다. Adaptive mode에서 지원하지 않는 조합은 명시적 오류를 낸다.

Stage 2가 논문대로 자동 실행된다고 가정하지 않는다. 공개 `main.py`는 `CustomTrainerStage1`을 선택하고 있으므로, 본 변경은 `stage != stage1`에서 adaptive mode를 거부한다. [S3]

**참조도는 teacher의 target-forced context에 대한 직접 attention proxy이다. Student가 실제로 사용한 정도나 인과적 기여도를 측정한다고 주장하지 않는다.**

## 3. 기존 코드 지도와 수정 경계

줄 번호는 기준 커밋의 파일 기준 1-based 참고값이다. 구현자는 커밋을 고정한 뒤 함수 이름으로 재확인한다.

| 기존 파일 / 심볼 | 역할 또는 주의점 | 조치 |
|---|---|---|
| `src/main.py::collate_fn_stage1`, 약 27–82행 | latent placeholder와 입력/보조 이미지 batch 구성 | 반환 데이터 형식 유지 |
| `src/main.py::main_train`, 약 86–230행 | config, model, trainer 연결 | 새 설정 전달, seed와 실행 manifest 연결 |
| `src/utils_deepseed.py::get_args` | CLI 인자 | history 설정과 runtime 검증 설정 추가 |
| `src/trainer.py::CustomTrainerStage1._teacher_build_latents`, 약 266–378행 | target 생성 | 기존 구현을 legacy 분기로 보존, adaptive builder 연결 |
| `src/trainer.py::_user_side_attn_pooling` | global intent 구성 | 수식/가중 방식 유지. latent attention 추출 코드로 재사용하지 않음 |
| `src/trainer.py::_maybe_group_for_helper`, `_top_p_top_k`, `_build_firstK_mask` | candidate/selection/mask | 기존 알고리즘 유지 |
| `src/trainer.py::compute_loss`, 약 379–430행 | target 주입과 next-token alignment | 반환 계약 유지. adaptive 경로의 오류 은폐만 방지 |
| `src/trainer.py::_EMATeacher`, `optimizer_step` | EMA lifecycle | 실제 update 횟수 검증 후 공통 runtime 수정 |
| `src/task_deepseed.py` | 이미지 경로 해석 | 이미지 root를 환경변수로 지정 가능하게 함 |
| `run_training.sh` | 실행 예 | 기존 파일 대신 별도 adaptive 실행 스크립트 추가 |
| `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` | custom backbone | v1에서 변경하지 않고 직접 호출/관측 |

공개 wrapper의 `outputs.hidden_states`는 일반 Hugging Face의 layer tuple이 아니라 최종 hidden-state tensor이다. 또한 decoder는 FlashAttention 경로에서 `self_attn.forward_ulysses()` / `forward_normal()`을 직접 호출한다. 이 때문에 wrapper의 `hidden_states[-2]`를 penultimate layer로 해석하거나, `self_attn` 모듈 hook만 등록하는 방식은 사용하지 않는다. [S4]

## 4. 수학적 계약

### 4.1. 위치와 표현

시퀀스를 `X → T1 → Z1 → … → Ti → Zi`로 정의한다. `s_i`는 `Zi`의 `<|latent_start|>` 위치이다. Target `Zi`를 선택할 때 사용하는 teacher 입력은 `input_ids[:, :s_i]`에 대응하는 prefix다. 현재 latent 시작 토큰도 이 prefix에 포함하지 않는다.

`K_j*`는 과거 j번째 구간에 실제로 선택한 target 수이다. Attention/latent summary의 대상 집합은 다음과 같다.

\[
\mathcal P_i=\bigcup_{j<i}\{\text{j번째 구간의 첫 }K_j^*\text{개 latent-pad 위치}\}.
\]

선택되지 않은 나머지 latent-pad는 기존 sequence에 남지만, 시각적 이력 집합에는 넣지 않는다. 해당 위치의 기본 token embedding과 attention-mask 값은 그대로 유지한다.

`h_t`는 teacher의 final-normalized hidden state, `g_p = h_p`는 실제 target이 주입된 과거 latent 위치의 contextual hidden state이다. Teacher 입력에 주입하는 raw selected target `v_p`와 query 요약에 사용하는 `g_p`를 구별한다.

### 4.2. Discounted text summary

유효한 reasoning text 위치만 모아 각 구간을 요약한다.

\[
\ell_r=\frac{1}{|T_r|}\sum_{t\in T_r}h_t.
\]

기본 text mode는 `discounted_segments`다.

\[
d_i=\frac{\sum_{r\leq i,\,|T_r|>0}\gamma^{i-r}\ell_r}
{\sum_{r\leq i,\,|T_r|>0}\gamma^{i-r}},\quad 0\leq\gamma\leq1.
\]

분모가 0이면 text 항 자체를 생략한다. `gamma=0`에서는 현재 구간만 사용하며, 현재 구간이 비어 있으면 과거로 임의 fallback하지 않는다. `gamma=1`은 구간별 동일 가중 평균이며 전체 토큰 평균과는 다르다.

순차 구현은 다음 상태만 유지하면 된다.

\[
S_i=\gamma S_{i-1}+\mathbb 1[|T_i|>0]\ell_i,
\qquad D_i=\gamma D_{i-1}+\mathbb 1[|T_i|>0].
\]

`d_i=S_i/D_i`이며, 빈 구간에서도 단계 번호는 증가한다. 이 상태는 **현재 sample의 현재 teacher snapshot** 안에서만 유효하다. 다음 sample, 다음 `compute_loss`, EMA update 이후까지 재사용하지 않는다.

Ablation용 `local` mode는 `ell_i`만, `token_mean` mode는 1…i 구간의 유효 token을 동일 가중치로 평균한다. `token_mean`은 논문식 pooling에 가까운 대조군이지 전체 논문 구현의 재현을 뜻하지 않는다.

### 4.3. 현재 reasoning에서 과거 latent로 향하는 attention

\[
a_{i,p}=\frac{1}{|\mathcal L|N_h|T_i|}
\sum_{\ell\in\mathcal L}\sum_{h=1}^{N_h}\sum_{t\in T_i}A^{\ell,h}_{t,p},
\quad p\in\mathcal P_i.
\]

`A`는 각 query 위치에서 **모든 유효 causal key를 분모로 한 softmax**이다. Latent key만 모아 softmax하면 안 된다. Layer별/head별 확률을 계산한 뒤 평균하며, logits를 먼저 평균한 뒤 softmax하지 않는다.

\[
m_i=\sum_{p\in\mathcal P_i}a_{i,p},\qquad
b_i=\sum_{p\in\mathcal P_i}a_{i,p}g_p.
\]

`Ti` 또는 `P_i`가 비어 있으면 `a`는 빈/영벡터, `m_i=0`, `b_i=0`이다. 정상적인 확률과 mask 아래 `0 <= m_i <= 1`이다.

설명을 위해 `w_p=a_p/m_i`를 기록할 수 있지만, 실제 구현은 `b_i = a @ g`로 계산한다. 작은 `m_i`를 정규화해 없애거나 uniform weight로 대체하지 않는다. `m_i`는 confidence가 아니라 해당 context에서의 attention mass다.

Attention row는 현재 text token 위치 `t`의 state에 대한 것이다. 그 token의 생성 기여도를 평가하려고 `t-1`로 이동시키지 않는다. Student의 alignment loss shift와 혼동하지 않는다.

### 4.4. 최종 query

기존 global-intent 계산으로 얻은 `u`를 유지한다. Text 항이 존재하면 `v_i=1`, 없으면 0이다.

\[
q_i^*=\frac{u+v_i d_i+b_i}{1+v_i+m_i}.
\]

일반적인 경우 분모는 `2+m_i`다. Cosine similarity에서는 이 공통 양의 분모를 생략해도 ranking이 같지만, 구현과 로그는 위 정의를 따른다. 각 항의 개별 L2 정규화, 추가 learnable gate, query 전체에 대한 discount는 도입하지 않는다.

FP32 누적 후 query가 nonfinite이거나 norm이 `1e-8` 이하이면 명시적 오류로 처리한다. 임의 target을 선택하지 않는다.

### 4.5. Candidate 선택과 loss

`_maybe_group_for_helper`와 `_top_p_top_k`를 그대로 사용한다.

\[
s_{i,j}=\cos(q_i^*,c'_{i,j}),\quad
J_i=\operatorname{ExistingTopPTopK}(s_i,\text{coverage_p},K).
\]

`candidate[J_i]`를 선택된 순서대로 해당 구간의 첫 `K_i*`개 pad에 대응시킨다. Spatial 순서로 다시 정렬하지 않는다. `K_i*`가 바뀔 수 있으므로 `teacher_latents`와 `firstK_mask`를 동일한 ledger로 만든다.

Student에는 기존 selected-target teacher forcing과 alignment를 적용한다. 예측 위치는 target 위치 바로 이전(`h[t-1]` 대 target at `t`)이다. CE/sim weight와 기존 target-noise 경로는 공통으로 유지한다. [S2]

## 5. Teacher context 구성과 실행 순서

### 5.1. 반드시 필요한 변경

이미 선택된 target을 teacher에 넣지 않으면 attention이 의미하는 것은 학습된 visual content의 참조가 아니라 placeholder 위치의 참조일 수 있다. 따라서 adaptive mode는 선택과 prefix 구성을 순차화한다.

1. 완성된 collated sample에서 구조·mask·helper image 수를 검증한다.
2. 입력 이미지의 visual embedding과 helper별 candidate pool을 teacher snapshot당 계산한다. Helper image는 candidate extractor에만 전달한다.
3. Global `u`를 기존 helper로 한 번 계산한다. 입력은 assistant header까지의 prompt prefix로 제한한다. 기존 pooling 수식과 fallback은 바꾸지 않는다.
4. `i=1…M`에 대해 `s_i` 직전 prefix embedding을 만든다. 입력 이미지와 **이미 선택된** j<i의 raw target만 삽입한다.
5. Teacher backbone을 실행하고 현재 text 및 과거 active latent의 final hidden state를 얻는다. 필요한 Q/K projection만 관측한다.
6. `ell_i`, `d_i`, `a_i`, `m_i`, `b_i`, `q_i*`를 계산한다.
7. i번째 helper candidate에서 target을 선택하고 ledger에 저장한다. 다음 단계 prefix에 이 target을 주입한다.
8. 모든 target을 sequence 위치 순으로 합쳐 기존 반환 계약으로 돌려준다.

현재/미래 helper image를 language-model context에 넣지 않는다. 미래 helper를 후보 계산용으로 미리 encode하는 것은 가능하지만 그것이 현재 query나 과거 target에 영향을 주어서는 안 된다. Image별 vision masking과 candidate slicing을 검사한다.

### 5.2. Backbone 호출

새 adapter는 `teacher.get_input_embeddings()`, `teacher.visual`, `teacher.get_rope_index(...)`, `teacher.model(...)`을 조합한다. Token embedding을 clone하여 입력 image feature와 과거 target을 원래 위치에 삽입한다.

Backbone 호출 계약:

```text
teacher.model(
    input_ids=None,
    inputs_embeds=prefix_embeds,
    attention_mask=prefix_attention_mask,
    position_ids=prefix_mrope_position_ids,
    use_cache=False,
    output_hidden_states=False,
    output_attentions=False,
    return_dict=True,
    generate_mode=False,
)
```

이 경로는 `last_hidden_state`를 직접 얻고, vocabulary-size logits를 불필요하게 생성하지 않는다. Layer tuple이 필요 없으므로 모든 layer hidden state를 반환하도록 설정하지 않는다.

MRoPE position은 해당 prefix의 **원본 입력 image grid**로 계산한다. 단순한 1D `arange`로 대체하지 않는다. `input_ids`와 `inputs_embeds`를 backbone에 동시에 전달하지 않는다. 새 adapter의 embedding 조립 결과는 기존 wrapper의 selected-target 주입 결과와 작은 fixture에서 일치해야 한다.

Teacher는 `eval()`, `requires_grad=False`, `torch.no_grad()`로 실행한다. `labels`, `pixel_values_latent`, `image_grid_thw_latent`를 teacher language backbone으로 전달하지 않는다. 원래 `inputs` dict와 tensor를 제자리 수정하지 않는다.

### 5.3. 첫 단계와 빈 text 구간

첫 단계에는 과거 latent가 없으므로 attention collector를 실행하지 않는다. `Ti`가 빈 경우에도 target 생성은 가능하다. 과거 text의 discounted summary가 존재하면 그것과 `u`를 사용하고, text summary도 없으면 `u`만 사용한다. 빈 구간을 assistant header나 latent boundary token으로 채우지 않는다.

## 6. Attention 수집 구현

### 6.1. 방식: FlashAttention 유지 + 필요한 probability만 재구성

전체 `output_attentions=True`를 켜는 방식을 production 경로로 쓰지 않는다. 선택된 layer의 실제 Q/K projection 출력을 관측하여 현재 text → 과거 latent probability만 계산한다.

Q/K projection에는 실제 layer normalization과 projection bias/weight가 반영되어 있어야 한다. Global-intent helper의 근사 scoring 로직을 복사하면 안 된다. 다음 순서를 따른다.

- 선택한 **decoder layer**에 `register_forward_pre_hook(with_kwargs=True)`를 붙여 `position_embeddings=(cos,sin)`을 확보한다.
- 같은 layer의 **`q_proj`와 `k_proj`**에 forward hook을 붙여 실제 projection 출력을 얻는다.
- 원본 `apply_multimodal_rotary_pos_emb` 및 GQA head mapping을 적용한다.
- `QK^T/sqrt(head_dim)`에 원래의 causal/padding 조건을 적용한다.
- 전체 유효 key에 대한 softmax 분모를 구한 뒤 과거 active latent key의 probability만 합산한다.

Decoder가 attention의 별도 forward 메서드를 직접 호출해도 그 안의 projection module 호출은 관측할 수 있다. `self_attn` 객체의 forward hook만 등록하는 설계는 금지한다. [S4]

### 6.2. Mask와 dtype

Key는 `key_position <= query_position`이고 원래 `attention_mask=1`인 위치만 허용한다. 현재 text 자신의 위치도 포함한다. 초기 이미지, prompt, 과거 text, 일반 special token 및 inactive latent-pad도 실제 context의 유효 key이면 softmax 분모에 포함한다. 다만 `a_i`의 출력 대상은 active visual latent만이다.

v1은 `use_sliding_window=False`, cache 미사용, 2D padding mask만 지원한다. Sliding-window/packed/block-diagonal mask가 들어오면 오류를 내며 full attention으로 조용히 대체하지 않는다.

모델 계산은 기존 BF16 설정을 유지하고, attention logits/softmax denominator 및 평균 누적은 FP32로 계산한다. 재구성 확률은 동일한 수학적 attention의 FP32 평가이며 FlashAttention kernel과 bitwise 동일함을 보장하는 주장은 하지 않는다.

### 6.3. Chunking

기본 chunk: query 64, key 2048, query-head 4. Layer별로 관측 직후 집계하고 projection cache를 해제한다.

각 head/query chunk에 대해:

1. 모든 key block의 masked log-sum-exp를 `logaddexp`로 누적한다.
2. Active latent key block에 대해서만 `exp(selected_logit - log_denominator)`를 계산한다.
3. Query/head 방향으로 합쳐 길이 `|P_i|`의 accumulator에 더한다.
4. 모든 선택 layer 종료 후 `|L| * N_h * |Ti|`로 나눈다.

`[all_layers, all_heads, sequence_length, sequence_length]`나 full attention matrix를 저장하지 않는다. Query 또는 head를 샘플링하지 않고 모두 계산한다. Hook은 context manager로 등록하고 정상 종료/예외 모두에서 `finally`로 제거한다. Global-intent 계산과 student forward에는 collector가 활성화되어 있으면 안 된다.

Prefix의 전체 key 길이를 `N_i`라고 할 때 추가 attention 계산량은 단계당 대략 `O(|L| N_h |Ti| N_i head_dim)`이다. Active latent가 적어도 softmax 분모 계산에는 전체 causal key가 필요하다. 전체 teacher prefix 재실행 비용도 존재하므로 속도 개선을 보장하지 않는다. v1의 선택은 정확성과 검증 가능성을 우선한 것이다.

## 7. 데이터·인터페이스 계약

### 7.1. 입력

기존 collator 필드를 유지한다: `input_ids [1,S]`, `attention_mask [1,S]`, `labels [1,S]`, `pixel_values`, `image_grid_thw`, `pixel_values_latent`, `image_grid_thw_latent`, `image_out_mask`.

길이 S의 위치는 **collator의 `process_batch` 적용 후** 좌표다. 원본 텍스트 문자열의 character offset으로 latent 위치를 추정하지 않는다.

Text span은 assistant header 뒤부터 첫 latent start까지, 이후에는 직전 latent end 다음부터 현재 latent start 전까지다. Header 패턴은 tokenizer로 인코딩하고 긴 패턴을 먼저 매칭한다. Header의 `assistant` 역할 문자열을 reasoning에 포함하지 않는다. Padding, image/latent special token, tokenizer special ID는 pooling 대상에서 제외한다. Ordinary text로 토큰화된 내용에 임의 문자열 필터를 추가하지 않는다.

### 7.2. 새 타입과 함수의 공개 계약

다음은 구현해야 할 인터페이스이며 현재 저장소에 이미 있다는 뜻이 아니다.

```python
from dataclasses import dataclass
from torch import Tensor

@dataclass(frozen=True)
class SegmentLayout:
    index: int                       # 0-based segment index
    start: int                       # latent_start position
    end: int                         # latent_end position, inclusive
    pad_positions: tuple[int, ...]
    text_positions: tuple[int, ...]

@dataclass(frozen=True)
class SampleLayout:
    assistant_content_start: int
    segments: tuple[SegmentLayout, ...]

@dataclass(frozen=True)
class LatentRecord:
    segment_index: int
    positions: tuple[int, ...]        # selected first-K positions
    target_embeddings: Tensor        # [Kstar, H], detached raw targets
    candidate_indices: tuple[int, ...]

@dataclass(frozen=True)
class PrefixObservation:
    last_hidden_state: Tensor        # [1, prefix_length, H]
    latent_positions: tuple[int, ...] # same order as attention_scores
    attention_scores: Tensor         # [M], FP32; empty if no active latents

@dataclass(frozen=True)
class QueryResult:
    query: Tensor                    # [1, H], FP32
    latent_mass: Tensor              # scalar FP32
    text_present: bool
    diagnostics: dict[str, float]

@dataclass(frozen=True)
class TargetBuildResult:
    latents: Tensor | None           # [1, sum(Kstar), H]
    first_k_mask: Tensor             # [1, S], bool
    records: tuple[LatentRecord, ...]
    diagnostics: tuple[dict, ...]
```

주요 API:

```python
parse_sample_layout(input_ids, attention_mask, tokenizer, latent_size) -> SampleLayout
summarize_text(segment_means, segment_lengths, mode, gamma) -> Tensor | None
aggregate_latent(hidden, scores, mode, previous_raw, permutation=None) -> tuple[Tensor, Tensor]
compose_query(global_intent, text_summary, latent_sum, latent_mass) -> QueryResult
TeacherPrefixAdapter.observe(inputs, layout, segment_index, records, collect_attention) -> PrefixObservation
HistoryTargetBuilder.build(inputs, k, p) -> TargetBuildResult
```

`hidden`은 `[M,H]`, `scores`는 `[M]`이며 순서가 동일해야 한다. `aggregate_latent` 반환은 `[1,H]`의 latent 합과 scalar mass다. `previous_raw`는 `last_raw` 대조군에서만 사용한다. `permutation`은 `attention_shuffled`에서만 사용하는 `[M]` LongTensor이며 trainer가 별도 generator로 생성한다. 이 mode에서 permutation이 없으면 오류다.

`TeacherPrefixAdapter`의 생성자에는 teacher, tokenizer, history config를 넘긴다. `HistoryTargetBuilder`에는 adapter와 기존 global-intent/candidate/selector/mask 함수를 callback으로 주입한다. 새 모듈이 `trainer.py`를 역으로 import하지 않도록 한다.

### 7.3. 반환과 student 연결

`CustomTrainerStage1._teacher_build_latents`의 외부 반환은 기존 `(latents, firstK_mask)` 그대로 유지한다. 내부 adaptive builder의 추가 diagnostics는 trainer가 보관하고 model forward 인자로 넘기지 않는다.

필수 invariant:

```text
latents.shape[1] == firstK_mask.sum() == sum(record.positions의 길이)
모든 selected position은 latent_pad이고 attention_mask=1
선택 순서 == raw target 주입 순서 == alignment target 순서
각 구간 Kstar <= config.latent_size
```

Adaptive mode에서 invariant 위반은 즉시 오류다. 기존 코드처럼 target mismatch를 CE-only training으로 조용히 우회하면 안 된다. Latent 구간이 전혀 없는 정상 sample만 CE-only 경로를 허용한다.

## 8. 파일 단위 변경 계획

기존 flat `src/` 구조를 유지한다. 아래 새 파일명은 제안 사항이 아니라 본 명세의 구현 대상 경로이다.

| 파일 | 생성/수정 | 책임 |
|---|---|---|
| `src/history_types.py` | 생성 | Config/dataclass와 공통 오류 타입 |
| `src/history_layout.py` | 생성 | Token/segment 위치 파싱과 구조 검증 |
| `src/history_query.py` | 생성 | Discount, latent aggregate, query fusion의 순수 tensor 함수 |
| `src/history_attention.py` | 생성 | Q/K 관측, MRoPE/GQA 적용, chunked probability 집계 |
| `src/history_teacher.py` | 생성 | Prefix embedding 조립, 순차 target ledger, builder |
| `src/ema_runtime.py` | 생성 | 검증된 EMA callback 및 teacher checkpoint lifecycle |
| `src/trainer.py` | 수정 | legacy/adaptive 분기, 새 builder 연결, 오류·로그·EMA 연결 |
| `src/utils_deepseed.py` | 수정 | CLI 설정 및 검증 |
| `src/main.py` | 수정 | Config 전달, seed, manifest, resume 경로 연결 |
| `src/task_deepseed.py` | 수정 | `ILVR_TRAIN_IMAGE_ROOT` / `ILVR_TEST_IMAGE_ROOT` 환경변수 우선 사용 |
| `run_training_history.sh` | 생성 | Adaptive 실행 예 및 필수 경로 검증 |
| `requirements-dev.txt` | 생성 | 테스트 전용 `pytest>=8,<9` |
| `tests/test_history_*.py` | 생성 | 아래 unit/integration 계약 |
| `tests/test_ema_runtime.py` | 생성 | EMA·resume·분산 검증 |
| `tests/conftest.py`, `pytest.ini` | 생성 | 작은 fixture, import path, GPU marker |
| `tests/zero3_history_smoke.py` | 생성 | torchrun으로 실행하는 2-GPU end-to-end 검증 |
| `docs/history_query_experiments.md` | 생성 | 대조군별 설정·실행·평가 및 측정 결과 기록 |
| `README.md` | 수정 | 새 기능의 실행법·지원 범위·검증 절 추가 |

`transformers/` 소스와 `eval.py`는 수정 대상이 아니다. Teacher adapter를 추가하기 위해 student model output schema를 바꾸지 않는다.

## 9. 설정 계약

### 9.1. CLI와 기본값

| CLI | 기본값 | 허용값 / 의미 |
|---|---|---|
| `--history_query_mode` | `legacy` | `legacy`, `adaptive` |
| `--history_text_mode` | `discounted_segments` | `local`, `token_mean`, `discounted_segments` |
| `--history_text_gamma` | `0.8` | 유한한 실수, 0…1 |
| `--history_latent_mode` | `attention_mass` | `last_raw`, `uniform_hidden`, `attention_mass`, `attention_conditional`, `attention_shuffled`, `none` |
| `--history_attention_layers` | `last` | `last`, `last4`, `all` |
| `--history_query_chunk_size` | `64` | 양의 정수 |
| `--history_key_chunk_size` | `2048` | 양의 정수 |
| `--history_head_chunk_size` | `4` | 양의 정수 |
| `--history_log_steps` | `20` | optimizer-step 기준, 양의 정수 |
| `--ema_runtime_mode` | `upstream` | `upstream`, `verified` |
| `--seed` | `42` | 기존 hard-coded seed를 CLI로 노출 |
| `--resume_from_checkpoint` | `None` | 재개할 기존 checkpoint 디렉터리. 경로가 없으면 오류 |

`HistoryQueryConfig` 필드명은 CLI의 `history_` 접두어를 제외한 `query_mode`, `text_mode`, `text_gamma`, `latent_mode`, `attention_layers`, `query_chunk_size`, `key_chunk_size`, `head_chunk_size`, `log_steps`로 고정한다. Runtime mode와 seed는 run manifest에 별도로 저장한다.

Adaptive mode는 `ema_runtime_mode=verified`, `stage=stage1`, `use_lora=False`, batch size 1, 지원 backbone/mask를 요구한다. `last4`에서 decoder layer가 4개 미만이면 오류다. `local`/`token_mean`에서는 gamma를 사용하지 않으며 이 사실을 로그에 기록한다.

기본 `legacy + upstream`은 기존 selector 및 EMA 경로를 유지한다. `legacy + verified`는 selector를 유지하되 공통 EMA lifecycle이 수정된 별도 대조군이다. 두 경우를 동일한 원본 재현으로 표시하지 않는다.

### 9.2. Latent ablation mode의 정확한 의미

- `last_raw`: 직전 nonempty selected raw feature 평균, mass=1. 없으면 0.
- `uniform_hidden`: 모든 과거 active latent hidden state 평균, mass=1. 없으면 0.
- `attention_mass`: 본 제안의 `b=a@g`, `m=sum(a)`.
- `attention_conditional`: `m>0`일 때 `(a/m)@g`, fusion mass=1. 모두 0이면 latent 항 생략.
- `attention_shuffled`: `a`를 past active position 사이에서 permute하되 합과 분포를 유지, `b=a_perm@g`.
- `none`: `b=0`, `m=0`.

Shuffle은 별도 local generator를 사용하여 student loss의 RNG stream을 변경하지 않는다. Seed는 `SHA256(seed, optimizer_step, microstep_index, rank, input_ids_bytes, segment_index)` 앞 8 byte를 little-endian 정수로 변환하고 `2**63-1`로 모듈로 연산하여 얻는다. Python의 process-randomized `hash()`는 사용하지 않는다.

### 9.3. 기존 설정 유지

`latent_size=8`, `coverage_p=0.9`, `helper_group_L=256`, `image_pool_k=8`, `sim_weight=1.0`, `ce_weight=1.0`, `ema_tau=0.999`를 첫 비교의 공통 설정으로 유지한다. 이는 검사한 코드 설정에 맞춘 기준이지 모든 데이터셋의 논문 설정과 동일하다는 뜻은 아니다. [S3, S5]

## 10. EMA와 재현성: 방법 변경과 분리할 공통 선행 작업

### 10.1. 왜 별도 gate인가

공개 subclass에는 `optimizer_step` override가 있지만, 함께 제공된 Trainer loop는 `self.optimizer.step()` 뒤 callback을 호출한다. 그 loop에서 subclass의 해당 메서드를 호출하는 경로는 확인되지 않았다. 따라서 현재 EMA가 의도대로 갱신된다고 전제해서는 안 된다. 정적 검토상의 위험이며 실제 실행에서 update count를 검증한다. [S2, S6]

### 10.2. Verified runtime 요구사항

`on_optimizer_step` callback을 명시적으로 등록한다. 실제 성공한 optimizer update마다 EMA를 정확히 한 번 수행하고, gradient-accumulation microstep/평가/target build에서는 수행하지 않는다. Optimizer step이 overflow 등으로 skip되면 갱신하지 않는다. Callback과 기존 override가 중복으로 update하지 않도록 한다.

Teacher는 student와 storage를 공유하지 않는 frozen replica여야 한다. ZeRO-3 partition 이후의 빈 parameter를 그대로 deepcopy하지 않는다. 기존 초기화 위치에서 full-shaped parameter 복사가 가능한지 확인하고, 불가능하면 training 시작 전 teacher를 올바르게 materialize한다.

ZeRO-3에서는 모든 rank가 동일한 parameter-name 순서로 collective에 참여하여 student parameter를 읽고, 각 rank의 teacher replica를 갱신한다. Rank 0의 teacher만 갱신해서는 안 된다. Name/shape 일치를 검사하며 `zip(parameters())` 순서에만 의존하지 않는다.

Teacher dtype과 EMA decay는 모든 비교군에서 동일하게 유지한다. FP32 master teacher로 바꾸는 별도 변경은 이번 실험에 섞지 않는다.

### 10.3. 저장·재개

Verified mode의 checkpoint에 `ema_teacher.safetensors`, `ema_state.json`, `history_query_config.json`, `run_manifest.json`을 저장한다. `ema_state.json`은 decay, 성공 update 수, checkpoint global step을 포함한다. v1 checkpoint는 optimizer-step 경계에서만 저장하며 microstep counter는 0이어야 한다.

Resume 시 첫 target build 전에 teacher와 config를 복구한다. Student만 복원하고 teacher를 처음 상태로 두거나 student로 다시 초기화하는 것을 정상 resume으로 취급하지 않는다. Adaptive resume에서 teacher 파일 또는 config가 없으면 오류다. 기존 checkpoint를 `--model`로 읽어 새 학습을 시작하는 경우는 명시적 warm start이며 optimizer/RNG resume이 아니다. `src/main.py`는 `trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)`로 전달하고, teacher 복원은 해당 checkpoint의 첫 target build 이전에 완료한다.

`run_manifest.json`에는 base commit, 현재 구현 commit, dirty 여부, 패키지 버전/`transformers.__file__`, 모델·tokenizer revision, 데이터 파일 SHA256, resolved layer indices, seed, precision, world size, gradient accumulation, runtime mode, 전체 history 설정을 기록한다.

## 11. 오류·경계 조건

| 조건 | Adaptive mode의 요구 동작 |
|---|---|
| Latent 구간 0개 | 기존 CE-only 학습, 빈 target mask |
| Latent 구간이 있으나 helper가 없음 | sample 식별 정보와 함께 오류 |
| Helper 수와 latent 구간 수 불일치 | 오류. 여분 helper를 무시하지 않음 |
| Unmatched/nested latent marker, pad 개수 불일치 | 학습 전에 오류 |
| Empty current reasoning | attention mass 0, text 규칙은 §4.2 준수 |
| `Kstar < K` | 선택 위치만 ledger에 기록. 나머지 pad를 가짜 visual feature로 집계하지 않음 |
| Left/right padding | 유효 위치의 query/선택 결과는 동등. 출력 mask는 원래 좌표 |
| 중간에 구멍이 있는 padding mask/packed sequence | v1에서는 오류 |
| Query/feature/attention에 NaN/Inf | 즉시 오류. CE-only fallback 금지 |
| Candidate pool이 비었음 | 해당 구간/helper 정보와 함께 오류 |
| Image feature 수와 image-token 수 불일치 | 새 adapter에서 오류. 임의 crop/균등 배분 금지 |
| batch>1, LoRA, unsupported backbone, Stage 2 adaptive | 명시적 NotImplementedError/ValueError |
| Attention capture 누락 또는 중복 | 오류. Uniform fallback 금지 |
| Hook 내부 또는 teacher forward 예외 | 모든 hook/cache 해제 후 예외 전달 |

`strict`를 끄는 연구용 우회 옵션은 v1에 제공하지 않는다.

## 12. 테스트 및 수용 기준

### 12.1. Unit test: 순수 tensor와 layout

| 테스트 이름 | 필수 assertion |
|---|---|
| `test_discount_two_segments` | `ell1=(1,0), ell2=(0,1), gamma=.5` → `(1/3,2/3)` |
| `test_discount_three_segments` | 독립 기저 3개, gamma=.5 → `(1/7,2/7,4/7)` |
| `test_gamma_zero_is_local` | 현재 nonempty이면 local과 일치. empty면 text 항 없음 |
| `test_gamma_one_is_segment_not_token_mean` | 길이가 다른 구간에서 두 평균을 구분 |
| `test_empty_segment_keeps_age` | 빈 구간도 discount exponent의 단계에 포함 |
| `test_attention_mass_not_renormalized` | `a=(.001,.002)` → `m=.003`, `b=.001*g1+.002*g2` |
| `test_full_context_denominator` | valid key 4개, logits 동일, 그중 latent 2개 → 각 .25, mass=.5 |
| `test_no_latents_and_no_text` | 최종 query=u, NaN 없음 |
| `test_query_denominator_cosine_invariance` | 양의 공통 분모 유무에 cosine 동일 |
| `test_layout_excludes_headers_and_specials` | assistant 역할 문자열과 pad가 text span에 없음 |
| `test_layout_padding_and_malformed` | 좌/우 padding 처리, malformed 입력은 오류 |
| `test_variable_kstar_ledger` | 여러 구간에서 target 순서와 mask sum 일치 |

### 12.2. Attention 정확성

작은 random-initialized Qwen2.5-VL language backbone fixture로 검증한다. Eager attention의 native probability를 reference로 사용한다. Head 수와 KV-head 수가 다른 GQA, 텍스트/이미지 MRoPE, left/right padding을 포함한다.

`test_qk_probe_matches_eager`: FP32 fixture에서 `atol=1e-6, rtol=1e-5`. BF16 fixture에서는 우선 `atol=5e-3, rtol=2e-2`를 수용 기준으로 사용하고 최대/평균 오차를 기록한다. 이 수치는 제안한 테스트 허용치이며 측정 결과가 아니다.

`test_chunked_matches_dense`: 여러 query/key/head chunk 크기에서 동일한 reduced score.

`test_last4_and_all_average_probabilities`: layer별 native probability 평균과 일치.

`test_projection_hooks_fire_on_flash_special_forward`: `forward_ulysses`/`forward_normal` 경로에서도 projection capture가 수행되고 각 선택 layer가 한 번씩 집계된다.

`test_hooks_removed_after_exception`: 예외 후 teacher/student forward에 hook이 남지 않는다.

### 12.3. Causality와 injection

**가장 중요한 통합 테스트는 다음 두 가지다.**

`test_future_suffix_does_not_change_current_query`: teacher와 prefix를 고정하고 Zi 이후의 text·latent placeholder·미래 helper를 변경한다. `q_i`, `a_i`, 현재 target index가 허용 오차 안에서 동일해야 한다. 현재 helper만 변경하면 `q_i`는 동일하고 candidate/target은 달라질 수 있다.

`test_previous_targets_are_actually_injected`: i>1에서 teacher backbone이 받은 `inputs_embeds`를 capture하고, 과거 selected pad 위치가 ledger의 raw target과 동일한지 검사한다. 실제 특징 대신 placeholder embedding이 남으면 실패다.

`test_adapter_matches_wrapper_embedding_injection`: input image와 selected target을 포함한 adapter embedding이 기존 wrapper의 해당 injection 규칙과 일치한다.

실제 model에서 과거 target 값을 바꾸었을 때 결과가 항상 크게 변해야 한다는 통계적 주장은 테스트 조건으로 강제하지 않는다. 값이 올바르게 주입되고 synthetic 의존성 fixture에서 정보 흐름이 달라지는 것을 검증한다.

### 12.4. Trainer, gradient, EMA, resume

`test_legacy_selector_parity`: 동일한 fixed teacher와 RNG에서 기존 selector와 legacy 분기의 target/mask가 일치한다. Verified EMA가 켜진 전체 학습까지 원본과 bitwise 동일하다는 뜻은 아니다.

`test_student_alignment_shift`: target 위치 t와 `pred_h[t-1]`가 맞물린다. `test_teacher_no_grad_student_has_grad`에서는 teacher parameter grad가 없고 student의 유효 trainable parameter grad가 존재한다.

`test_ema_once_per_successful_optimizer_step`: accumulation=2, microstep=4에서 EMA update=2. Skip step fixture를 추가한다. FP32 scalar fixture에서 teacher=1, student=3, tau=.9이면 갱신 결과=1.2.

`test_ema_resume_matches_uninterrupted`: 동일한 seed의 CPU/tiny-model deterministic run에서 uninterrupted와 save/resume의 teacher state 및 다음 target이 일치한다.

`test_zero3_teacher_replicas_match`: 2 GPU integration에서 EMA parameter checksum/update count가 모든 rank에서 동일하다. Teacher parameter가 비거나 sharded placeholder가 아니어야 한다.

### 12.5. 성능·완료 기준

단계별 prefix 길이와 Ti 길이에 대해 teacher latency, attention-probe latency, peak allocated VRAM, target build 횟수를 기록한다. 전체 attention matrix 저장이 없음을 확인한다. 장문 fixture는 8k에서 검사하고, 실제 학습 최대 길이(설정상 최대 32k)는 사용 하드웨어에서 별도 검증한다. 메모리 예산과 처리량은 측정값으로 보고하며 사전 보장하지 않는다.

구현 완료는 unit/reference/causality/gradient/EMA 테스트와 지원 환경의 smoke run 통과를 뜻한다. **방법의 유효성은 별도 controlled experiment로 판단하며, 테스트 통과를 성능 개선의 증거로 쓰지 않는다.**

## 13. 실험 비교와 로그

### 13.1. 최소 비교군

| ID | Teacher/context | Text | Latent | 목적 |
|---|---|---|---|---|
| R0 | untouched snapshot + upstream runtime | 기존 local | 기존 last raw | 공개 코드 실행 기준 |
| R1 | verified runtime + sequential injected context | local | last_raw | 공통 context/구조 정비 효과 |
| C0 | R1과 동일 | local | uniform_hidden | 모든 active hidden history를 쓰는 공통 대조군 |
| C1 | R1과 동일 | discounted_segments, gamma=.8 | uniform_hidden | Discount 효과 |
| C2 | R1과 동일 | local | attention_mass | Attention 효과 |
| C3 | R1과 동일 | discounted_segments, gamma=.8 | attention_mass | 제안 전체 |
| C4 | R1과 동일 | C3와 동일 | attention_shuffled | 내용에 맞춘 weight 배정 효과 |

R1에는 새 text-span 경계 정리도 포함되므로 R0 대비 차이를 attention 자체의 성과로 해석하지 않는다. C0/C1/C2/C3의 2×2 비교가 핵심이다. Gamma=1 대조군과 `attention_conditional`은 각각 segment pooling과 mass 보존 효과를 추가 분리한다.

Gamma 탐색은 `{0, .5, .8, .95, 1}`로 제한해 시작한다. Layer 설정 `last/last4/all`은 동일한 데이터 분할과 seed에서 비교한다. Validation에서 설정을 정하고 test는 최종 평가에만 사용한다. 반복 실험 seed는 `{42,43,44}`로 통일한다.

`coverage_p=.9`에서는 실제 Kstar 분포가 바뀔 수 있다. 모든 군의 Kstar 분포를 보고한다. “동일한 실제 K” 비교를 주장하려면 공통으로 `coverage_p=1`을 사용하고 candidate 수가 K 이상인지 확인한 별도 실험을 수행한다.

### 13.2. 필수 로그

Optimizer-step 단위 scalar: `latent_mass_mean`, `active_latent_count`, `selected_kstar_mean`, `current_text_token_count`, `query_norm`, `global_norm`, `text_summary_norm`, `latent_summary_norm`, `text_latest_coefficient`, `teacher_forward_ms`, `attention_probe_ms`, `peak_memory_bytes`, `ema_update_count`.

Attention entropy/최대 상대 weight는 `m_i>0`일 때만 정의하여 기록한다. NaN, fallback, missing-capture는 정상 scalar가 아니라 error다. 모든 단계의 Ti/position/selection debug dump는 기본 비활성화하고 테스트 또는 소수 sample 진단에만 사용한다. 원문 reasoning/image bytes를 기본 로그에 저장하지 않는다.

평가는 같은 checkpoint-selection 규칙, seed, generation budget, evaluator로 수행한다. Target index의 변화량은 선택이 바뀌었다는 진단이지 target 품질 향상의 직접 증거가 아니다.

## 14. 실행 계약

아래 명령은 **구현 완료 후 사용할 예시**이며 지금 저장소에서 이미 지원하는 새 CLI라는 뜻이 아니다.

```bash
# 저장소 고정과 custom Transformers 경로 확인
 git checkout 13a67ed2ee975d6ca56b4e9eba18d68b352b166b
 python -c "import transformers; print(transformers.__version__, transformers.__file__)"

# DATA_JSONL은 디렉터리가 아니라 실제 JSONL 파일이어야 한다.
export ILVR_TRAIN_IMAGE_ROOT="${IMAGE_ROOT:?Set IMAGE_ROOT to the image root}"

accelerate launch --num_processes="${NUM_PROCESSES:?Set NUM_PROCESSES}" src/main.py \
  --model "Qwen/Qwen2.5-VL-7B-Instruct" \
  --stage stage1 --task zebra-cot \
  --data_path "${DATA_JSONL:?Set DATA_JSONL to TRAIN.jsonl}" \
  --save_model_path "${OUTPUT_DIR:?Set OUTPUT_DIR}" \
  --log_file "${LOG_FILE:?Set LOG_FILE}" \
  --cache_dir "${HF_HOME:?Set HF_HOME}" \
  --epochs 15 --gradient_accumulation_steps 8 \
  --latent_size 8 --coverage_p 0.9 --helper_group_L 256 \
  --image_pool_k 8 --sim_weight 1.0 --ce_weight 1.0 \
  --warm_up_steps 10 --save_steps 612 --ema_tau 0.999 \
  --history_query_mode adaptive --ema_runtime_mode verified \
  --history_text_mode discounted_segments --history_text_gamma 0.8 \
  --history_latent_mode attention_mass --history_attention_layers last \
  --seed 42
```

환경은 repository의 Python 3.11, custom Transformers 설치 절차를 따른다. 저장소의 `requirements.txt`에는 Torch/FlashAttention을 별도로 설치하는 주석이 있으므로 의존성이 완비되었다고 가정하지 않는다. 검사한 fork의 version 문자열은 `4.50.0.dev0`이다. [S1, S7, S8]

단위 테스트 실행 계약:

```bash
python -m pytest -q tests -m "not gpu and not distributed"
python -m pytest -q tests/test_history_attention.py -m gpu
python -m pytest -q tests/test_history_integration.py -m gpu
torchrun --standalone --nproc_per_node=2 tests/zero3_history_smoke.py
```

마지막 smoke script는 새 테스트 산출물이다. 지원하지 않는 GPU 환경에서 skip한 테스트를 통과했다고 보고하지 않는다.

## 15. 최종 acceptance checklist

- [ ] 기준 커밋과 import 중인 custom Transformers 경로가 manifest에 기록된다.
- [ ] `legacy + upstream` 분기의 기존 selector 동작이 보존된다.
- [ ] Adaptive teacher가 실제 선택된 과거 target을 prefix에 주입한다.
- [ ] Current/future latent와 미래 reasoning을 query에 사용하지 않는다.
- [ ] Attention은 full causal context 분모를 사용하고 MRoPE/GQA/mask를 반영한다.
- [ ] Last/last4/all 설정이 native reference와 수치적으로 검증된다.
- [ ] 작은 attention mass가 강제 정규화되어 사라지지 않는다.
- [ ] Variable Kstar와 target↔position↔loss shift가 일치한다.
- [ ] Teacher gradient 없음, student gradient 있음, hook 누수 없음이 검증된다.
- [ ] EMA가 성공 optimizer step마다 한 번 갱신되고 resume에 보존된다.
- [ ] 2-GPU ZeRO-3 smoke와 지원 길이의 메모리 검증 결과가 기록된다.
- [ ] 동일한 공통 pipeline에서 text/latent 변경의 독립적 효과를 비교할 수 있다.

## 16. 근거와 조회 범위

아래는 원문 소스를 확인하기 위한 참조이다. 코드의 line range는 §3을 기준으로 하고, 링크는 가능하면 동일 immutable commit을 사용한다. 이 명세의 신규 함수·파일·기본 실험값·테스트 허용치는 제안한 구현 계약이며 원 repository에 존재한다고 주장하지 않는다.

[S0] ILVR arXiv v3, §3.2–3.3, Eq. (4)–(6): https://arxiv.org/html/2512.05665v3

[S1] Repository README와 기준 commit: https://github.com/XD111ds/ILVR ; https://github.com/XD111ds/ILVR/commit/13a67ed2ee975d6ca56b4e9eba18d68b352b166b

[S2] Target selection / EMA / loss: https://github.com/XD111ds/ILVR/blob/13a67ed2ee975d6ca56b4e9eba18d68b352b166b/src/trainer.py

[S3] Collator / model loading / trainer wiring: https://github.com/XD111ds/ILVR/blob/13a67ed2ee975d6ca56b4e9eba18d68b352b166b/src/main.py

[S4] Custom Qwen2.5-VL: https://github.com/XD111ds/ILVR/blob/13a67ed2ee975d6ca56b4e9eba18d68b352b166b/transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py

[S5] CLI / data paths: https://github.com/XD111ds/ILVR/blob/13a67ed2ee975d6ca56b4e9eba18d68b352b166b/src/utils_deepseed.py ; https://github.com/XD111ds/ILVR/blob/13a67ed2ee975d6ca56b4e9eba18d68b352b166b/src/task_deepseed.py

[S6] Vendored Trainer optimizer/callback loop: https://github.com/XD111ds/ILVR/blob/13a67ed2ee975d6ca56b4e9eba18d68b352b166b/transformers/src/transformers/trainer.py

[S7] Dependencies: https://github.com/XD111ds/ILVR/blob/13a67ed2ee975d6ca56b4e9eba18d68b352b166b/requirements.txt

[S8] Fork version: https://github.com/XD111ds/ILVR/blob/13a67ed2ee975d6ca56b4e9eba18d68b352b166b/transformers/src/transformers/__init__.py

검토 한계: 공개 source를 웹에서 읽어 정적 검토했다. 로컬 git clone은 네트워크 제한으로 수행하지 못했으며 모델 가중치/데이터 로드, dependency 설치, training, GPU profiling을 실행하지 않았다. 따라서 runtime 위험을 확인된 장애 또는 해결 완료로 표현하지 않는다.
