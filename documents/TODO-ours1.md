# ILVR 기반 Step-wise Modality DPO — Codex 구현 명세

> **대상:** Codex 기반 구현 작업
> 
> 
> **Base:** `XD111ds/ILVR`의 정상 동작하는 Stage 1/Stage 2 환경
> 
> **목표:** ILVR의 interleaved reasoning에 **genuine Step-wise DPO + modality-ablation entropy regularization**을 추가하고, correctness 확보 후 shared-prefix/tree-packed computation으로 최적화한다.
> 
> **구현 참고:** `huggingface/trl`(DPO), `JIA-Lab-research/Step-DPO`(step granularity), `frankxwang/dpo-prefix-sharing`(prefix sharing/tree attention)
> 

---

## 1. 고정된 방법론

Reasoning unit:

$$
T_{k-1}\rightarrow V_k=(I_k^{\mathrm{helper}},Z_k)\rightarrow T_k
$$

- $T_{k-1}$: 직전 text reasoning block
- $I_k^{\mathrm{helper}}$: visual reasoning source
- $Z_k$: ILVR latent visual representation
- $T_k$: DPO 및 entropy의 scoring unit
- $P_{k-1}$: $T_{k-1}$보다 이전의 clean history

Clean context:

$$
C_k^+=(Q,I_0,P_{k-1},T_{k-1},Z_k)
$$

Visual-off:

$$
C_k^{-V}
=(Q,I_0,P_{k-1},T_{k-1},\operatorname{Ablate}(Z_k))
$$

Text-off:

$$
C_k^{-T}
=(Q,I_0,P_{k-1},\operatorname{Ablate}(T_{k-1}),Z_k)
$$

본 방법이 학습하는 것은 total causal effect가 아니라 **clean history가 고정된 상태에서 직전 component의 marginal contribution**이다.

첫 구현의 ablation은 attention-edge masking으로 고정한다.

```
V-off:
  query      = T_k span
  blocked KV = Z_k span

T-off:
  query      = T_k span
  blocked KV = T_{k-1} span
```

$Q$, $I_0$, $P_{k-1}$ 및 나머지 clean history는 변경하지 않는다. Token/latent를 물리적으로 삭제하지 않는다.

---

## 2. Training lifecycle

```
ILVR Stage 1
    ↓
Stage 2A
    - 기존 ILVR Stage 2 CE
    - self-generated latent 안정화
    ↓
Stage 2A snapshot
    ├── frozen DPO reference
    └── offline V/T-off negative generation
    ↓
Stage 2B
    - clean CE
    - Step-wise DPO
    - ablated-branch entropy regularization
```

Stage 2A/2B 때문에 기존 Stage 2 optimizer-step budget을 두 배로 늘리지 않는다.

$$
N_{\mathrm{Stage2}}
=
N_{2A}+N_{2B}
$$

Stage 2A 종료 checkpoint:

$$
\pi_{\mathrm{ref}}
=
\pi_{\mathrm{Stage2A}}
$$

`stage2a_ratio`는 config로 노출한다.

---

## 3. 데이터 계약

기존 `sequence_plan`의 reasoning-step identity를 collator 이후까지 보존한다.

```python
@dataclass
class ReasoningStepSpan:
    step_id: int
    prev_text_start: int
    prev_text_end: int
    latent_start: int
    latent_end: int
    target_text_start: int
    target_text_end: int
    helper_image_idx: int | None
```

모든 span은 half-open interval `[start, end)`이다.

Invariant:

```
prev_text_span  == T_{k-1}
latent_span     == Z_k
target_text_span== T_k
```

Collator는 기존 ILVR tensor 외에 최소 다음을 반환한다.

```python
{
    # existing ILVR fields ...
    "reasoning_step_spans": list[list[ReasoningStepSpan]],
    "sample_ids": list[str],
}
```

DPO/entropy target에는 `T_k`의 assistant text token만 포함한다. Latent control token, padding, unrelated position은 제외한다.

---

## 4. Offline negative generation

Stage 2A frozen checkpoint에서 modality-off reasoning을 생성한다.

$$
T_{k,V}^{-}
\sim
\pi_{\mathrm{Stage2A}}(\cdot\mid C_k^{-V})
$$

$$
T_{k,T}^{-}
\sim
\pi_{\mathrm{Stage2A}}(\cdot\mid C_k^{-T})
$$

Chosen은 clean target $T_k^+$이다.

Stage 2B training 중 online generation하지 않는다.

권장 저장 schema:

```json
{
  "sample_id": "sample-001",
  "step_id": 2,
  "chosen_text": "...",
  "rejected_visual_text": "...",
  "rejected_textual_text": "...",
  "valid_visual_negative": true,
  "valid_textual_negative": true
}
```

첫 버전 filtering:

1. chosen과 exact-match인 negative 제외
2. whitespace normalize 후 동일한 negative 제외
3. empty/min-length 미달 negative 제외
4. 기존 dataset에 이미 correctness checker가 있으면 명백히 correct인 negative 제외
5. 새로운 judge dependency는 첫 버전에 추가하지 않음

Invalid DPO pair여도 entropy branch 계산은 가능하다.

---

## 5. Genuine Step-wise DPO

Block score:

$$
s_\theta(T_k\mid C)
=
\frac{1}{n_k}
\sum_{j=1}^{n_k}
\log\pi_\theta(t_{k,j}\mid C,t_{k,<j})
$$

Ablated context는 **negative 생성**에 사용한다. DPO scoring은 같은 clean context $C_k^+$에서 chosen/rejected를 비교한다.

$$
T_k^+\succ T_{k,V}^{-}
\quad\text{under } C_k^+
$$

$$
T_k^+\succ T_{k,T}^{-}
\quad\text{under } C_k^+
$$

Modality $m\in\{V,T\}$에 대해:

$$
\Delta_{\theta,k}^{m}
=
s_\theta(T_k^+\mid C_k^+)
-
s_\theta(T_{k,m}^{-}\mid C_k^+)
$$

$$
\Delta_{\mathrm{ref},k}^{m}
=
s_{\mathrm{ref}}(T_k^+\mid C_k^+)
-
s_{\mathrm{ref}}(T_{k,m}^{-}\mid C_k^+)
$$

$$
\boxed{
\mathcal L_{\mathrm{DPO},k}^{m}
=
-\log\sigma
\left[
\beta
(
\Delta_{\theta,k}^{m}
-
\Delta_{\mathrm{ref},k}^{m}
)
\right]
}
$$

전체 Step-DPO는 valid pair만 평균한다.

TRL의 DPO numerical-stability 및 reference-logprob 처리 방식을 참고하되 **ILVR trainer를 TRL `DPOTrainer`로 교체하지 않는다.**

---

## 6. Entropy regularization

Clean distribution:

$$
p_{k,j}^{+}
=
\pi_\theta(\cdot\mid C_k^+,t_{k,<j})
$$

Ablated distribution:

$$
p_{k,j}^{-m}
=
\pi_\theta(\cdot\mid C_k^{-m},t_{k,<j})
$$

Shannon entropy:

$$
H(p)=-\sum_{v\in\mathcal V}p(v)\log p(v)
$$

Absolute entropy maximization 대신 clean-relative margin을 사용한다.

$$
\boxed{
\mathcal L_{H,k}^{m}
=
\frac{1}{n_k}
\sum_{j=1}^{n_k}
\left[
\gamma_H
-
\left(
H(p_{k,j}^{-m})
-
H(p_{k,j}^{+})
\right)
\right]_+
}
$$

즉 다음 정도까지만 uncertainty 증가를 요구한다.

$$
H(p_{k,j}^{-m})
\ge
H(p_{k,j}^{+})+\gamma_H
$$

Entropy는 `target_text_span`에서만 계산한다. 선택된 logits에서 `log_softmax` 후 즉시 entropy를 reduce하여 불필요한 full-probability tensor 저장을 피한다.

---

## 7. Stage 2B 최종 objective

$$
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{CE}}^{\mathrm{clean}}
+
\lambda_D\mathcal L_{\mathrm{StepDPO}}
+
\lambda_H\mathcal L_{\mathrm{Entropy}}
}
$$

초기 config:

```yaml
step_preference:
enabled:true
beta:0.1
lambda_dpo:1.0
lambda_entropy:0.1
entropy_margin:0.1
ablation: attention_mask
reference: stage2a
```

수치는 초기 실험값이며 방법론의 고정 상수가 아니다.

---

## 8. Attention-ablation API

```python
@dataclass
class StepAblationSpec:
    sample_idx: int
    step_id: int
    mode: Literal["visual", "text"]
    query_start: int
    query_end: int
    blocked_kv_start: int
    blocked_kv_end: int
```

생성 규칙:

```
visual:
  query      = target_text_span[k]
  blocked_kv = latent_span[k]

text:
  query      = target_text_span[k]
  blocked_kv = prev_text_span[k]
```

Mask는 기존 causal mask에 추가된다. 다른 visibility는 변경하지 않는다.

Qwen2.5-VL/ILVR의 logical position 및 multimodal RoPE semantics는 clean branch와 동일하게 유지한다.

---

## 9. 구현 전략: naive → packed

### Phase A — Naive correctness implementation

최적화하지 않는다. 독립 forward로 정확한 reference loss를 만든다.

필요 computation:

```
policy:
  clean chosen/rejected DPO scores
  clean entropy logits
  V-off entropy logits
  T-off entropy logits

reference:
  clean chosen/rejected DPO scores
```

필수 logging:

```
loss/ce
loss/dpo_visual
loss/dpo_text
loss/entropy_visual
loss/entropy_text

stats/dpo_margin_visual
stats/dpo_margin_text
stats/entropy_clean
stats/entropy_visual_off
stats/entropy_text_off
```

### Phase B — Shared-prefix/tree-packed optimization

Phase A가 검증된 뒤 `dpo-prefix-sharing`의 구조를 참고해 최적화한다.

Target graph:

```
clean trajectory
T0 ─ Z1 ─ T1 ─ Z2 ─ T2 ─ ...
          │           │
          ├─ V-off    ├─ V-off
          └─ T-off    └─ T-off
```

Packed implementation 요구사항:

1. clean prefix 계산 공유
2. sidecar는 자신의 clean ancestor와 자기 branch 내부 이전 token만 attend
3. 해당 V/T edge 추가 차단
4. sidecar 간 attention 금지
5. logical position ID 유지
6. 필요한 clean-prefix gradient 보존
7. 가능하면 block-sparse/FlexAttention으로 실제 masked computation도 감소

단순히 forward API 호출 수만 줄이는 것은 optimization 완료로 간주하지 않는다.

---

## 10. 파일 책임

기존 ILVR 구조를 최대한 유지하고 새 기능은 모듈화한다.

```
src/
├── main.py
├── trainer.py
├── task_deepseed.py
└── step_preference/
    ├── spans.py
    ├── negatives.py
    ├── ablation.py
    ├── dpo_loss.py
    ├── entropy_loss.py
    └── packing.py

scripts/
└── generate_step_negatives.py

tests/
└── step_preference/
    ├── test_spans.py
    ├── test_ablation_mask.py
    ├── test_dpo_loss.py
    ├── test_entropy_loss.py
    └── test_packed_equivalence.py
```

책임:

- `spans.py`: `sequence_plan` → `ReasoningStepSpan`
- `negatives.py`: offline pair schema/loading/filter
- `ablation.py`: V/T visibility mask
- `dpo_loss.py`: block score + genuine Step-DPO
- `entropy_loss.py`: relative entropy-margin
- `packing.py`: Phase B에만 사용
- `trainer.py`: 위 모듈을 조합하는 orchestration만 담당

---

## 11. 필수 테스트

### 11.1 Span correctness

Synthetic input에서 $T_{k-1}$, $Z_k$, $T_k$가 정확히 추적되는지 검증.

### 11.2 Ablation isolation

V-off에서 정확히

$$
T_k\not\rightarrow Z_k
$$

만 차단되고 이전 clean history visibility는 보존되는지 검사. T-off도 동일.

### 11.3 DPO numerical equivalence

Toy logits에서 custom DPO loss가 TRL의 표준 DPO 식과 tolerance 내에서 동일해야 한다.

### 11.4 Entropy loss

수동 distribution에서

$$
H(p^{-m})-H(p^+)\ge\gamma_H
$$

이면 entropy loss가 0이어야 한다.

### 11.5 Baseline preservation

`step_preference.enabled=false`이면 기존 ILVR Stage 2 forward/loss semantics가 변하지 않아야 한다.

### 11.6 Naive vs packed

Phase B 완료 조건:

$$
|\mathcal L_{\mathrm{naive}}-\mathcal L_{\mathrm{packed}}|
<
\epsilon_L
$$

그리고 대표 trainable parameter subset에서

$$
\frac{
\|\nabla\mathcal L_{\mathrm{naive}}
-
\nabla\mathcal L_{\mathrm{packed}}\|_2
}{
\|\nabla\mathcal L_{\mathrm{naive}}\|_2+\epsilon
}
<
\epsilon_G
$$

를 만족해야 한다.

---

## 12. Codex 실행 순서

아래 순서를 변경하지 않는다.

```
1. 기존 ILVR Stage 2 baseline test 확인
2. reasoning step span metadata 추가
3. offline negative generation 구현
4. Step-DPO loss + unit test
5. entropy-margin loss + unit test
6. naive V/T-off forward 구현
7. Stage 2B에서 CE + DPO + entropy 통합
8. small-subset overfit/sanity test
9. tree-packed/shared-prefix 구현
10. naive vs packed loss/gradient equivalence 검증
11. throughput / peak-memory benchmark
```

1–8이 통과하기 전에 9를 시작하지 않는다.

---

## 13. 첫 버전의 비목표

다음은 core implementation 이후 별도 extension으로 둔다.

- zero/null latent replacement
- latent Gaussian perturbation/dropout
- helper-image source perturbation
- global V/T ablation
- context-calibrated DPO
- reference-free SimPO
- adaptive entropy margin
- step-type별 modality weighting
- online negative regeneration

---

## 14. 완료 기준

다음 조건을 모두 만족하면 첫 구현 완료로 본다.

1. `step_preference.enabled=false`에서 기존 ILVR baseline 보존
2. 모든 valid sample에서 $T_{k-1}\rightarrow Z_k\rightarrow T_k$ span 추적
3. Stage 2A checkpoint로 V/T-off negative offline 생성
4. Stage 2B에서 clean CE + Step-DPO + entropy 학습
5. V/T-off가 지정된 attention edge만 제거
6. 모든 loss component와 margin/entropy statistic logging
7. naive implementation sanity training 성공
8. packed loss/gradient가 naive와 tolerance 내 일치
9. packed version의 wall-clock 또는 memory 개선 확인

구현 우선순위는 항상 다음과 같다.

$$
\boxed{
\text{Correctness}
\;>\;
\text{Methodological fidelity}
\;>\;
\text{Efficiency}
}
$$
