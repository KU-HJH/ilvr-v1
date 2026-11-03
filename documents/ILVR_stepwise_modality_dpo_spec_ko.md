# ILVR 기반 Step-wise Modality DPO 및 Entropy Regularization 방법론 명세서

> **문서 상태:** 구현 전 방법론 명세 초안  
> **기준 코드베이스:** ILVR  
> **주요 학습 목표:** interleaved reasoning의 각 전이에서 text/visual modality의 국소적 기여를 step 단위로 학습하고, 필요한 modality가 제거된 조건에서는 predictive uncertainty가 증가하도록 정렬한다.

---

## 1. 요약

본 방법은 ILVR의 interleaved reasoning 구조

$$
T_{k-1}
\rightarrow
V_k=(I_k^{\mathrm{helper}}, Z_k)
\rightarrow
T_k
$$

에서 **다음 text reasoning block $T_k$를 학습 및 평가 단위**로 사용한다.

- $T_{k-1}$: 직전 text reasoning block
- $I_k^{\mathrm{helper}}$: visual reasoning step의 원천이 되는 helper image
- $Z_k$: helper image의 visual semantics를 내재화한 ILVR latent representation
- $T_k$: 직전 text/visual reasoning state를 이용해 생성되는 다음 text reasoning block

각 $T_k$에 대해 다음 세 조건을 구성한다.

$$
C_k^{+}: \text{clean context}
$$

$$
C_k^{-V}: \text{직전 visual state } Z_k \text{의 접근을 차단한 context}
$$

$$
C_k^{-T}: \text{직전 text state } T_{k-1} \text{의 접근을 차단한 context}
$$

그 후 다음 두 목적을 결합한다.

1. **Step-wise DPO:** clean context에서 올바른 reasoning step이 modality ablation으로 생성된 잘못된 reasoning step보다 선호되도록 학습한다.
2. **Entropy regularization:** 필요한 modality가 제거된 branch에서는 clean branch보다 next-token predictive entropy가 일정 margin 이상 증가하도록 학습한다.

최종 Stage 2B objective는 다음과 같다.

$$
\boxed{
\mathcal L_{\mathrm{S2B}}
=
\mathcal L_{\mathrm{CE}}^{\mathrm{clean}}
+
\lambda_D\mathcal L_{\mathrm{StepDPO}}
+
\lambda_H\mathcal L_{\mathrm{Entropy}}
}
$$

이 학습은 clean trajectory를 backbone으로 유지하고 각 reasoning step에서 짧은 counterfactual sidecar branch를 분기하는 방식으로 계산한다. 초기 구현은 독립 forward를 사용하는 정확성 기준 구현으로 만들고, 이후 shared-prefix tree packing으로 최적화한다.

---

## 2. 연구 문제

### 2.1 Sequence-level DPO의 preference saturation

기존 sequence-level DPO는 전체 response의 preference margin을 하나로 합산한다.

$$
\mathcal L_{\mathrm{seq}}
=
-\log\sigma\left(
\beta\sum_{k=1}^{K}\Delta_k
\right)
$$

일부 쉬운 text token, 초기 reasoning step 또는 final-answer token이 큰 양의 margin을 만들면 전체 sigmoid가 먼저 포화될 수 있다. 이 경우 아직 preference가 제대로 학습되지 않은 다른 reasoning step도 공통적으로 작은 gradient를 받는다.

따라서 본 방법은 preference objective를 reasoning block별로 분리한다.

$$
\mathcal L_{\mathrm{step}}
=
\sum_{k=1}^{K}
-\log\sigma(\beta\Delta_k)
$$

각 step의 preference margin이 독립적으로 최적화되므로, 한 step의 포화가 다른 step의 학습을 차단하지 않는다.

### 2.2 Multimodal shortcut

전체 response 수준의 multimodal preference는 쉬운 textual cue나 final answer만으로 충족될 수 있다. 그 결과 중간 visual reasoning step이 실제로 visual evidence에 의존하는지 보장되지 않는다.

본 방법의 질문은 다음과 같다.

> **현재 reasoning transition에서 직전 text/visual reasoning state가 실제로 필요한가?**

이를 위해 clean history를 유지한 상태에서 현재 $T_k$가 직전 $T_{k-1}$ 또는 $Z_k$를 보지 못하도록 국소적으로 intervention한다.

### 2.3 Modality 제거 후 과신

DPO는 특정 rejected response의 likelihood를 낮출 수 있지만, 제거된 probability mass가 다른 하나의 shortcut response에 집중되는 것을 막지 못한다.

예를 들어

$$
(0.8, 0.1, 0.1)
\rightarrow
(0.1, 0.8, 0.1)
$$

은 정답 likelihood는 감소하지만 모델은 여전히 과신한다. 따라서 ablated branch의 **full predictive distribution entropy**를 직접 제어하는 별도 regularization이 필요하다.

---

## 3. 범위와 비범위

### 3.1 본 방법이 직접 학습하는 것

본 방법은 다음 quantity를 학습한다.

> 이전 clean reasoning history가 고정됐을 때, 직전 text block 또는 visual latent block이 현재 text reasoning block에 제공하는 **국소적·주변적 기여(marginal contribution)**.

### 3.2 본 방법이 직접 주장하지 않는 것

현재 초안은 다음을 직접 추정하지 않는다.

- 전체 trajectory 수준의 total causal effect
- 이전 text에 이미 매개된 visual information의 완전한 제거
- 이전 latent에 이미 매개된 text information의 완전한 제거
- 모든 modality 경로의 causal decomposition

따라서 결과는 `total modality dependence`가 아니라 `step-local marginal dependence`로 해석한다.

---

## 4. 표기

하나의 interleaved reasoning trajectory를 다음과 같이 정의한다.

$$
\tau
=
(Q, I_0, T_0, V_1, T_1, V_2, T_2, \ldots, V_K, T_K)
$$

여기서

- $Q$: instruction 또는 question
- $I_0$: initial input image
- $I_k^{\mathrm{helper}}$: $k$번째 visual reasoning step의 helper image
- $Z_k$: $I_k^{\mathrm{helper}}$의 semantics를 내재화한 ILVR latent representation
- $V_k=(I_k^{\mathrm{helper}}, Z_k)$: 개념적 visual reasoning step
- $T_k=(t_{k,1},\ldots,t_{k,n_k})$: $k$번째 text reasoning block
- $P_{k-1}$: $T_{k-1}$ 이전의 모든 clean history

현재 step의 clean context는 다음과 같다.

$$
C_k^{+}
=
(Q,I_0,P_{k-1},T_{k-1},Z_k)
$$

중요한 점은 $T_k$가 prediction 및 preference scoring unit이라는 것이다. $Z_k$는 discrete vocabulary token이 아니므로 일반적인 token-level DPO score를 직접 부여하지 않는다. 대신 $Z_k$가 downstream $T_k$의 likelihood와 preference에 미치는 영향으로 latent를 학습한다.

---

## 5. Helper image와 latent의 역할

### 5.1 개념적 정의

Helper image는 visual modality의 일부이다.

$$
V_k=(I_k^{\mathrm{helper}}, Z_k)
$$

- $I_k^{\mathrm{helper}}$는 visual evidence의 source이다.
- $Z_k$는 그 evidence를 reasoning context에 전달하는 representation이다.

### 5.2 ILVR Stage 1에서의 역할

Stage 1에서는 helper image를 teacher가 처리하여 $Z_k$의 supervision target을 제공한다.

$$
I_k^{\mathrm{helper}}
\xrightarrow{\text{teacher}}
Z_k^{\mathrm{target}}
$$

학생 모델은 text CE와 latent alignment를 통해 visual latent semantics를 bootstrap한다.

### 5.3 ILVR Stage 2에서의 역할

Stage 2에서는 모델이 $Z_k$를 self-generated latent state로 사용한다. 현재 주 방법의 local V-off는 source image 자체보다 **$Z_k\rightarrow T_k$ 정보 경로**에 intervention한다.

따라서 다음을 구분한다.

#### Representation-level intervention — 주 방법

$$
T_k \not\leftarrow Z_k
$$

현재 text block이 clean $Z_k$의 K/V에 attend하지 못하도록 막는다.

#### Source-level intervention — 선택적 ablation

$$
I_k^{\mathrm{helper}}
\rightarrow
\widetilde I_k^{\mathrm{helper}}
\rightarrow
\widetilde Z_k
\rightarrow
T_k
$$

Helper image 자체를 mask, blur, crop 또는 perturb하여 visual source의 변화가 latent와 downstream reasoning에 미치는 영향을 측정한다.

현재 구현의 기본안은 representation-level intervention이며, helper-image perturbation은 후속 ablation study로 둔다.

---

## 6. Step-local counterfactual context

### 6.1 Clean context

$$
C_k^{+}
=
(Q,I_0,P_{k-1},T_{k-1},Z_k)
$$

### 6.2 Visual-off context

기본 V-off는 $T_k$ query가 직전 latent $Z_k$를 볼 수 없도록 attention edge를 차단한다.

$$
C_k^{-V}
=
(Q,I_0,P_{k-1},T_{k-1},\operatorname{Mask}_V(Z_k))
$$

토큰이나 latent slot 자체를 삭제하지 않고 visibility만 제거한다.

### 6.3 Text-off context

기본 T-off는 $T_k$ query가 직전 text block $T_{k-1}$을 볼 수 없도록 attention edge를 차단한다.

$$
C_k^{-T}
=
(Q,I_0,P_{k-1},\operatorname{Mask}_T(T_{k-1}),Z_k)
$$

질문 $Q$, initial image $I_0$, 그보다 이전의 clean history $P_{k-1}$은 유지한다.

### 6.4 Intervention 해석

$Z_k$는 $T_{k-1}$을 조건으로 생성될 수 있으므로 T-off branch에서도 text information 일부가 $Z_k$를 통해 남을 수 있다. 반대로 $T_{k-1}$에는 이전 visual information이 verbalize되어 있을 수 있다.

따라서

$$
C_k^{+} \text{ vs. } C_k^{-V}
$$

는 현재 transition에서 $Z_k$의 추가적인 marginal contribution을 측정하고,

$$
C_k^{+} \text{ vs. } C_k^{-T}
$$

는 현재 transition에서 $T_{k-1}$의 추가적인 marginal contribution을 측정한다.

---

## 7. Z-off operator

### 7.1 기본안: attention-edge masking

$$
A(q,j)=-\infty,
\qquad
q\in T_k,
\quad
j\in Z_k
$$

장점:

- sequence length가 유지된다.
- logical position ID가 유지된다.
- zero latent와 같은 OOD 입력을 만들지 않는다.
- clean $Z_k$를 그대로 유지하므로 intervention이 국소적이다.

### 7.2 선택적 대안

#### Null-latent replacement

$$
Z_k\rightarrow Z_{\mathrm{null}}
$$

$Z_{\mathrm{null}}$ 후보:

- zero latent
- dataset mean latent
- learned null latent

#### Latent perturbation

$$
\widetilde Z_k=Z_k+\epsilon
$$

또는 latent dropout, dimension shuffle, span shuffle 등을 사용할 수 있다.

#### Helper-image perturbation

$$
I_k^{\mathrm{helper}}
\rightarrow
\widetilde I_k^{\mathrm{helper}}
$$

blur, crop, region mask 등을 통해 source-level robustness를 측정한다.

기본 실험은 attention-edge masking으로 고정하고, 나머지는 intervention operator ablation으로 비교한다.

---

## 8. ILVR training stage와의 결합

### 8.1 Stage 1 — Perceptual bootstrap

ILVR Stage 1은 유지한다.

$$
\boxed{
\mathcal L_{\mathrm{S1}}
=
\mathcal L_{\mathrm{CE}}^{\mathrm{text}}
+
\lambda_{\mathrm{latent}}
\mathcal L_{\mathrm{latent-align}}
}
$$

목적은 helper image의 visual semantics를 $Z_k$에 bootstrap하는 것이다. Stage 1에는 Step-DPO와 entropy regularization을 적용하지 않는다.

### 8.2 Stage 2A — Latent relaxation warm-up

Stage 1 checkpoint에서 시작하여 self-generated latent reasoning으로 전환한다.

$$
\boxed{
\mathcal L_{\mathrm{S2A}}
=
\mathcal L_{\mathrm{CE}}^{\mathrm{clean}}
}
$$

Stage 2A는 독립적인 추가 학습 budget이 아니라 전체 Stage 2 optimizer step budget의 앞부분으로 둔다.

$$
N_{\mathrm{Stage2}}
=
N_{\mathrm{2A}}+N_{\mathrm{2B}}
$$

초기 기본안은 전체 Stage 2 budget의 짧은 앞부분을 2A로 사용하고, 나머지를 2B로 사용한다.

Stage 2A 종료 checkpoint를 frozen reference model로 사용한다.

$$
\boxed{
\pi_{\mathrm{ref}}
=
\pi_{\mathrm{Stage2A-end}}
}
$$

### 8.3 Stage 2B — Step-wise modality alignment

Stage 2B에서는 clean CE, Step-wise DPO, entropy regularization을 결합한다.

$$
\boxed{
\mathcal L_{\mathrm{S2B}}
=
\mathcal L_{\mathrm{CE}}^{\mathrm{clean}}
+
\lambda_D\mathcal L_{\mathrm{StepDPO}}
+
\lambda_H\mathcal L_{\mathrm{Entropy}}
}
$$

---

## 9. Step-wise negative 생성

### 9.1 Positive step

$T_k^{+}$는 dataset의 정답 reasoning block 또는 별도 검증을 통과한 preferred reasoning block이다.

### 9.2 Modality-specific rejected step

Stage 2A frozen model을 이용하여 modality-ablated context에서 negative를 offline 생성한다.

$$
T_{k,V}^{-}
\sim
\pi_{\mathrm{ref}}(\cdot\mid C_k^{-V})
$$

$$
T_{k,T}^{-}
\sim
\pi_{\mathrm{ref}}(\cdot\mid C_k^{-T})
$$

V-off negative는 직전 visual state가 보이지 않을 때 발생하는 reasoning error를 반영하고, T-off negative는 직전 textual reasoning state가 보이지 않을 때 발생하는 reasoning error를 반영한다.

### 9.3 Offline generation을 기본으로 하는 이유

- training 중 rollout 비용을 제거한다.
- negative가 계속 변하는 non-stationarity를 피한다.
- 동일 preference pair로 naive/packed implementation을 비교할 수 있다.
- reference score를 재사용하기 쉽다.

### 9.4 Negative filtering

다음 negative는 제거하거나 별도 표시한다.

- positive와 동일하거나 거의 동일한 응답
- 정답 또는 유효한 대안 reasoning으로 판정된 응답
- 빈 응답 또는 지나치게 짧은 응답
- 지나치게 긴 응답
- syntax 오류로 인해 trivial하게 구분되는 응답

최소 metadata:

```yaml
step_id: int
modality: visual | text
positive_text: str
negative_text: str
negative_valid: bool
negative_reason: str | null
```

---

## 10. Step-wise DPO formulation

### 10.1 Block-normalized log-likelihood

길이가 다른 reasoning block을 비교하기 위해 average token log-likelihood를 사용한다.

$$
\bar\ell_{\theta}(Y\mid C)
=
\frac{1}{|Y|}
\sum_{t=1}^{|Y|}
\log\pi_{\theta}(y_t\mid C,y_{<t})
$$

### 10.2 Reference-adjusted preference margin

각 modality $m\in\{V,T\}$에 대해

$$
\begin{aligned}
A_{\theta,k}^{m}(C)
=&\;
\bar\ell_{\theta}(T_k^{+}\mid C)
-
\bar\ell_{\theta}(T_{k,m}^{-}\mid C)
\\
&-
\left[
\bar\ell_{\mathrm{ref}}(T_k^{+}\mid C)
-
\bar\ell_{\mathrm{ref}}(T_{k,m}^{-}\mid C)
\right]
\end{aligned}
$$

를 정의한다.

### 10.3 Clean-context Step-DPO

Positive와 negative를 동일한 clean context에서 비교한다.

$$
\boxed{
\mathcal L_{\mathrm{resp},k}^{m}
=
-\log\sigma\left(
\beta A_{\theta,k}^{m}(C_k^{+})
\right)
}
$$

이는 standard response preference에 가장 가까운 genuine Step-DPO 항이다.

### 10.4 Context-calibrated Step-DPO

Full context에서의 preference margin이 modality-ablated context보다 커지도록 학습한다.

$$
\boxed{
\mathcal L_{\mathrm{cal},k}^{m}
=
-\log\sigma\left(
\beta
\left[
A_{\theta,k}^{m}(C_k^{+})
-
A_{\theta,k}^{m}(C_k^{-m})
-
\mu_D
\right]
\right)
}
$$

여기서 $\mu_D\ge 0$는 clean-vs-ablated preference-margin 차이에 요구하는 최소 margin이다.

### 10.5 전체 Step-DPO loss

$$
\boxed{
\mathcal L_{\mathrm{StepDPO}}
=
\frac{1}{\sum_{k,m}w_{k,m}}
\sum_{k=1}^{K}
\sum_{m\in\{V,T\}}
w_{k,m}
\left(
\mathcal L_{\mathrm{resp},k}^{m}
+
\lambda_C
\mathcal L_{\mathrm{cal},k}^{m}
\right)
}
$$

기본 초안에서는

$$
w_{k,V}=w_{k,T}=1
$$

로 시작한다. 이후 LOOK/THINK/FUSE 등의 step type을 사용할 경우 modality별 weight를 조정할 수 있으나, 이는 초기 구현 범위 밖의 후속 확장으로 둔다.

---

## 11. Entropy regularization

### 11.1 목적

DPO는 특정 negative response의 probability를 낮추지만 ablated branch의 전체 distribution shape를 직접 제어하지 않는다. Entropy 항은 필요한 modality가 제거된 branch의 과신을 억제한다.

### 11.2 Token-level predictive distribution

Positive reasoning block을 teacher-forcing하면서 각 token 위치의 full-vocabulary distribution을 계산한다.

$$
p_{k,t}^{+}
=
\pi_{\theta}(\cdot\mid C_k^{+},T_{k,<t}^{+})
$$

$$
p_{k,t}^{-m}
=
\pi_{\theta}(\cdot\mid C_k^{-m},T_{k,<t}^{+})
$$

Shannon entropy는

$$
H(p)
=
-\sum_{v\in\mathcal V}p(v)\log p(v)
$$

로 정의한다.

### 11.3 Relative entropy-margin loss

Ablated branch의 entropy가 clean branch보다 최소 $\gamma_H$ 이상 높아지도록 hinge loss를 사용한다.

$$
\boxed{
\mathcal L_{H,k}^{m}
=
\frac{1}{|T_k^{+}|}
\sum_{t=1}^{|T_k^{+}|}
\left[
\gamma_H
-
\left(
H(p_{k,t}^{-m})
-
H(p_{k,t}^{+})
\right)
\right]_{+}
}
$$

이 formulation은 ablated distribution을 vocabulary 전체의 완전한 uniform distribution으로 밀지 않고, clean branch보다 필요한 만큼만 더 uncertain하게 만든다.

### 11.4 전체 entropy loss

$$
\boxed{
\mathcal L_{\mathrm{Entropy}}
=
\frac{1}{\sum_{k,m}w_{k,m}}
\sum_{k=1}^{K}
\sum_{m\in\{V,T\}}
 w_{k,m}\mathcal L_{H,k}^{m}
}
$$

---

## 12. 최종 Stage 2B objective

$$
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{S2B}}
=&\;
\mathcal L_{\mathrm{CE}}^{\mathrm{clean}}
\\
&+
\lambda_D
\frac{1}{\sum_{k,m}w_{k,m}}
\sum_{k,m}
w_{k,m}
\left(
\mathcal L_{\mathrm{resp},k}^{m}
+
\lambda_C\mathcal L_{\mathrm{cal},k}^{m}
\right)
\\
&+
\lambda_H
\frac{1}{\sum_{k,m}w_{k,m}}
\sum_{k,m}
w_{k,m}\mathcal L_{H,k}^{m}
\end{aligned}
}
$$

각 항의 역할은 분리된다.

| 항 | 역할 |
|---|---|
| $\mathcal L_{\mathrm{CE}}^{\mathrm{clean}}$ | clean reasoning quality 보존 |
| $\mathcal L_{\mathrm{resp}}$ | 올바른 reasoning을 modality-specific negative보다 선호 |
| $\mathcal L_{\mathrm{cal}}$ | modality가 존재할 때 preference discrimination이 더 강해지도록 정렬 |
| $\mathcal L_H$ | modality가 제거된 branch의 과신 억제 |

---

## 13. Reference model

### 13.1 기본 선택

$$
\boxed{
\pi_{\mathrm{ref}}
=
\pi_{\mathrm{Stage2A-end}}
}
$$

이유:

- Stage 2의 self-generated latent regime과 동일하다.
- 현재 policy initialization과 가깝다.
- Stage 1 또는 base Qwen model보다 context distribution mismatch가 작다.
- 동일 checkpoint로 modality-specific negative를 생성할 수 있다.

### 13.2 Reference와 latent context 정렬

ILVR의 $Z_k$는 모델에 의해 생성되는 continuous latent이므로 policy와 reference가 독립적으로 $Z_k$를 만들면 서로 다른 context를 score하게 될 수 있다.

현재 ours1에서 보존하는 구현 계약은 independent-context comparison이다.

1. policy가 clean $Z_k^{\theta}$를 생성하고 policy sidecar들이 이를 공유한다.
2. frozen Stage 2A reference는 token/image 입력으로 자체 $Z_k^{\mathrm{ref}}$를 생성한다.
3. reference chosen/rejected score는 clean context에서 no-grad로 계산하여 offline scalar artifact에 저장한다.
4. Stage 2B는 저장된 scalar를 읽으며 현재 policy latent를 reference에 주입하지 않는다.

이전 초안의 $\operatorname{stopgrad}(Z_k^{\theta})$ reference 주입은 현재 구현이
아니다. 두 conditioning은 서로 다른 objective이므로 이번 계산 최적화에서
서로 교체하지 않는다. Reference-free objective도 추가하지 않는다.

Cache는 checkpoint, tokenizer/template, rendering, token/image/position 입력,
latent 생성 및 수치 설정이 고정된 경우에만 재사용한다. 기존 manifest는 외부
image byte 전체를 인증하지 않는다. 실제 입력 audit과 소규모 reference score
재계산을 별도 확인하며, invalid/missing cache는 소비하지 않는다. Reference
호출 0회는 frozen이라는 사실만으로 추정하지 않고 실제 online 호출과 유효
cache 소비를 구분하여 기록한다.

---

## 14. Clean trajectory와 counterfactual sidecar

### 14.1 논리적 구조

```text
Clean:
T0 ── Z1 ── T1 ── Z2 ── T2 ── ... ── ZK ── TK
          │             │                  │
          ├─ V-off T1   ├─ V-off T2       ├─ V-off TK
          └─ T-off T1   └─ T-off T2       └─ T-off TK
```

각 sidecar는 다음 원칙을 따른다.

1. 분기 지점 이전의 clean prefix를 공유한다.
2. current target block만 teacher-force한다.
3. sidecar 결과는 clean main path로 되돌아가지 않는다.
4. sidecar끼리는 서로 attend하지 않는다.
5. branch는 clean future token을 볼 수 없다.

### 14.2 Teacher forcing

- $Z_k$: Stage 2에서는 clean main path에서 self-generated된다.
- $T_k^{+}$ 및 $T_{k,m}^{-}$: log-likelihood 계산을 위해 branch suffix에서 teacher-force한다.
- clean CE는 positive clean trajectory에만 적용한다.
- sidecar에는 일반 CE를 적용하지 않고 DPO/entropy score 계산에만 사용한다.

### 14.3 Gradient 경로

Clean branch에서는

$$
\mathcal L
\rightarrow
T_k
\rightarrow
Z_k
\rightarrow
\theta
$$

gradient가 유지되어야 한다.

V-off branch는 $T_k\rightarrow Z_k$ edge가 차단되므로 해당 branch loss에서 $Z_k$로 직접 gradient가 흐르지 않는다. 이는 의도된 intervention이다.

Inference용 detached KV cache를 그대로 재사용하면 sidecar loss가 clean prefix parameter로 전달되는 gradient가 잘릴 수 있으므로, 최종 packed training은 하나의 autograd graph 안에서 수행해야 한다.

---

## 15. 계산 전략

2026-09-06 승인된 shared-prefix 확장은 clean/V/T 세 논리 그룹을 사용한다.
이는 실제 decoder forward 세 번의 보장이 아니다. Clean latent rollout의
내부 순차 호출, suffix chunk 수, checkpoint 재계산 및 reference 호출을
분리하여 계측한다. 기존 naive와 monolithic packed를 유지하며 새 경로의
동등성·비용 측정이 통과하기 전에는 완료나 성능 향상을 선언하지 않는다.

### 15.1 단계 1: Naive correctness implementation

각 branch를 독립적으로 계산한다.

- clean forward
- V-off forward
- T-off forward
- chosen/rejected score forward
- reference no-grad forward

이 구현은 느리지만 loss semantics의 기준이다.

### 15.2 단계 2: Shared-prefix local replay

Clean prefix의 계산 graph를 공유하고 각 $T_k$ branch suffix만 재계산한다.

### 15.3 단계 3: Tree-packed sidecar training

Clean trajectory와 모든 sidecar를 하나의 flattened tree sequence로 구성한다.

각 sidecar token은

- clean ancestor
- 동일 sidecar의 이전 token

만 attend할 수 있으며, 제거 대상 modality와 다른 sidecar는 보지 못한다.

실제 FLOPs 절감을 위해 dense boolean mask만 사용하는 것이 아니라 FlexAttention 또는 block-sparse attention을 적용한다.

### 15.4 Logical position 보존

Sidecar token의 position ID는 packed memory상의 물리적 위치가 아니라 독립 branch에서 가졌을 logical position을 사용한다.

예를 들어 clean $T_k$, V-off $T_k$, T-off $T_k$는 모두 같은 시작 position을 가져야 한다.

Qwen2.5-VL의 multimodal RoPE 및 vision position 처리와 충돌하지 않는지 별도 검증이 필요하다.

---

## 16. 데이터 및 collator 계약

ILVR의 `sequence_plan`에서 다음 metadata를 collator 이후까지 보존해야 한다.

```yaml
sample_id: str
initial_image_id: str
steps:
  - step_id: int
    text_prev_span: [start, end]
    helper_image_id: str
    latent_span: [start, end]
    text_target_span: [start, end]
    step_type: text_visual_text
    positive_text: str
    negative_visual_text: str | null
    negative_textual_text: str | null
    negative_visual_valid: bool
    negative_textual_valid: bool
```

Model input 단계에서는 최소 다음 mask가 분리되어야 한다.

- `clean_ce_mask`
- `positive_step_mask`
- `negative_step_mask`
- `latent_mask`
- `visual_ablation_mask`
- `text_ablation_mask`
- `branch_id`
- `logical_position_ids`

Sidecar positive/negative token에는 일반 clean CE를 적용하지 않는다.

---

## 17. 구현 구조

권장 모듈 구조는 다음과 같다.

```text
ILVR/
├── src/
│   ├── training/
│   │   ├── stage1_trainer.py
│   │   ├── stage2_trainer.py
│   │   └── step_dpo_trainer.py
│   ├── preference/
│   │   ├── step_segments.py
│   │   ├── negative_generation.py
│   │   ├── counterfactual.py
│   │   ├── step_dpo_loss.py
│   │   ├── entropy_loss.py
│   │   └── tree_packing.py
│   └── ...
└── transformers/
    └── .../qwen2_5_vl/
        └── modeling_qwen2_5_vl.py
```

### 모듈 책임

- `step_segments.py`: `sequence_plan`을 $T_{k-1}$, $Z_k$, $T_k$ span으로 변환
- `negative_generation.py`: Stage 2A reference로 V/T-off negative offline 생성 및 filtering
- `counterfactual.py`: V-off/T-off attention visibility specification 생성
- `step_dpo_loss.py`: response DPO와 context-calibrated DPO 계산
- `entropy_loss.py`: full-vocabulary relative entropy-margin 계산
- `tree_packing.py`: clean/sidecar tree flattening, mask, branch ID, logical position 생성
- `step_dpo_trainer.py`: Stage 2B orchestration

---

## 18. 필수 참고 저장소

### Base repository

- `XD111ds/ILVR`
- https://github.com/XD111ds/ILVR

ILVR의 data representation, latent reasoning, custom Qwen2.5-VL, Stage 1/2 training을 유지한다.

### DPO reference

- `huggingface/trl`
- https://github.com/huggingface/trl

Trainer 전체를 이식하지 않고 다음만 참고한다.

- policy/reference log-prob 계산
- length normalization
- DPO sigmoid loss
- reference model 처리

### Prefix/tree packing reference

- `frankxwang/dpo-prefix-sharing`
- https://github.com/frankxwang/dpo-prefix-sharing

다음을 참고하여 ILVR/Qwen2.5-VL에 맞게 재구현한다.

- prefix-sharing preprocessing
- branch-aware attention mask
- original position ID 보존
- branch별 log-prob extraction
- FlexAttention/block-sparse integration

### 선택적 개념 참고

- `JIA-Lab-research/Step-DPO`
- https://github.com/JIA-Lab-research/Step-DPO

현재 step을 preference unit으로 두는 데이터 semantics만 참고한다.

---

## 19. 구현 순서

### Milestone 1 — ILVR baseline 복원

1. Stage 1 reproduction
2. Stage 1 checkpoint에서 Stage 2 self-generated latent path 복원
3. 기존 ILVR Stage 2 성능 및 loss curve 확인

### Milestone 2 — Step metadata

1. `sequence_plan`의 step identity 보존
2. $T_{k-1}$, $Z_k$, $T_k$ span unit test
3. helper image와 latent mapping 검증

### Milestone 3 — Naive intervention

1. V-off attention masking
2. T-off attention masking
3. clean/V-off/T-off log-prob 및 entropy 계산
4. branch 간 context leakage 검증

### Milestone 4 — Offline negative dataset

1. Stage 2A reference snapshot
2. V-off/T-off negative generation
3. negative filtering
4. preference dataset serialization

### Milestone 5 — Naive Stage 2B loss

1. clean CE
2. Step-wise response DPO
3. context-calibrated DPO
4. relative entropy-margin
5. total loss integration

### Milestone 6 — Packed optimization

1. shared-prefix replay
2. tree packing
3. logical position/mRoPE 검증
4. block-sparse attention
5. naive와 loss/gradient equivalence 검증

---

## 20. 필수 검증 항목

### 20.1 Intervention correctness

- V-off에서 $T_k$가 $Z_k$를 attend하지 않는가?
- T-off에서 $T_k$가 $T_{k-1}$을 attend하지 않는가?
- question, initial image, earlier clean history는 유지되는가?
- sidecar가 clean future 또는 다른 sidecar를 보지 않는가?

### 20.2 Numerical equivalence

$$
\left|
\mathcal L_{\mathrm{naive}}
-
\mathcal L_{\mathrm{packed}}
\right|
<\epsilon_L
$$

$$
\frac{
\left\|
\nabla\mathcal L_{\mathrm{naive}}
-
\nabla\mathcal L_{\mathrm{packed}}
\right\|
}{
\left\|
\nabla\mathcal L_{\mathrm{naive}}
\right\|
}
<\epsilon_G
$$

### 20.3 Entropy behavior

- clean entropy가 불필요하게 증가하지 않는가?
- V-off/T-off entropy gap이 margin까지 증가하는가?
- entropy 증가가 특정 wrong answer로의 probability 이동이 아니라 distribution flattening으로 나타나는가?

### 20.4 Preference behavior

- 각 step의 positive-vs-negative margin이 독립적으로 개선되는가?
- 일부 step의 saturation이 다른 step의 gradient를 차단하지 않는가?
- clean-context margin이 ablated-context margin보다 커지는가?

### 20.5 Latent semantics

- Stage 2B 후에도 $Z_k$가 helper-image visual semantics를 유지하는가?
- clean task 성능 향상과 함께 latent alignment가 붕괴하지 않는가?
- 필요하면 weak latent-alignment regularizer를 추가할 수 있는가?

### 20.6 효율성

다음을 별도로 보고한다.

- optimizer step 수
- wall-clock time
- processed suffix token 수
- attention edge/FLOPs 추정
- peak GPU memory
- reasoning step 수 $K$에 따른 scaling

`forward call 1회`만으로 효율성을 주장하지 않는다.

---

## 21. 기본안과 선택적 대안

| 항목 | 기본안 | 선택적 대안 |
|---|---|---|
| Base repository | ILVR | 없음 |
| Preference unit | $T_k$ block | token-level 보조 분석 |
| Visual step | $(I_k^{helper}, Z_k)$ | raw image/latent 분리 분석 |
| V-off | $T_k\rightarrow Z_k$ attention mask | null latent, perturbation, helper-image perturbation |
| T-off | $T_k\rightarrow T_{k-1}$ attention mask | span dropout, null text state |
| Causal 해석 | local marginal contribution | global/mediated effect 후속 분석 |
| Negative 생성 | Stage 2A reference offline rollout | online refresh |
| Reference | Stage 2A frozen checkpoint | reference-free ablation |
| Entropy | clean-relative margin | absolute max entropy |
| 구현 | naive 후 packed | 처음부터 packed는 비권장 |

---

## 22. 현재 고정된 결정

다음은 구현 기본값으로 고정한다.

1. ILVR를 base repository로 사용한다.
2. reasoning unit은 $T_{k-1}\rightarrow V_k\rightarrow T_k$이다.
3. helper image와 latent를 하나의 visual reasoning step으로 해석한다.
4. preference scoring unit은 $T_k$이다.
5. 현재 intervention은 total effect가 아닌 local marginal contribution을 측정한다.
6. primary V-off는 $Z_k$ attention masking이다.
7. primary T-off는 $T_{k-1}$ attention masking이다.
8. Stage 2A frozen checkpoint에서 modality-specific negatives를 offline 생성한다.
9. Stage 2B는 clean CE, Step-wise DPO, context calibration, entropy regularization을 결합한다.
10. entropy는 absolute maximization이 아니라 clean-relative margin을 사용한다.
11. Stage 2A/2B는 전체 Stage 2 step budget을 분할하며, 자동으로 총 optimizer step을 두 배로 늘리지 않는다.
12. 구현은 naive correctness version을 먼저 만들고 tree-packed version으로 최적화한다.

---

## 23. 구현 전에 남은 검증 사항

방법론의 골격은 고정됐지만 다음은 코드 수준에서 검증해야 한다.

1. ILVR 공개 코드에서 Stage 2 self-generated latent path를 정확히 복원할 수 있는가?
2. Policy-generated $Z_k$를 frozen reference에 동일 context로 주입할 수 있는가?
3. Qwen2.5-VL mRoPE와 custom tree mask를 함께 사용할 수 있는가?
4. FlashAttention 2, gradient checkpointing, ZeRO-3와 packed tree attention이 호환되는가?
5. Naive와 packed implementation의 loss 및 gradient가 수치적으로 일치하는가?
6. Full-vocabulary entropy 계산의 메모리 비용을 감당할 수 있는가?

이 항목들은 방법론 재논의의 대상이라기보다 구현 feasibility 및 correctness test의 대상이다.

---

## 24. 최종 방법 정의

본 방법을 한 문장으로 정의하면 다음과 같다.

> **ILVR의 interleaved text–visual-latent–text reasoning에서 다음 text block을 preference unit으로 삼고, 직전 text 또는 visual latent의 접근을 국소적으로 제거하여 생성한 modality-specific rejected reasoning을 Step-wise DPO로 학습하며, 동시에 modality-ablated branch의 full predictive entropy가 clean branch보다 증가하도록 정규화한다.**

최종 핵심 수식은 다음과 같다.

$$
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{CE}}^{\mathrm{clean}}
+
\lambda_D
\sum_{k,m}
\left(
\mathcal L_{\mathrm{resp},k}^{m}
+
\lambda_C\mathcal L_{\mathrm{cal},k}^{m}
\right)
+
\lambda_H
\sum_{k,m}
\mathcal L_{H,k}^{m}
}
$$

여기서

$$
m\in\{V,T\}
$$

이며 clean trajectory의 여러 depth에서 발생하는 counterfactual branch는 shared-prefix tree packing을 통해 효율적으로 계산한다.
