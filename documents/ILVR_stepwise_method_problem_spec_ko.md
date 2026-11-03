# ILVR 기반 Step-wise Multimodal Preference 학습의 문제 정의 및 이슈 명세

> **문서 목적**  
> 본 문서는 현재 논의 중인 방법론의 구현 세부사항을 설명하기보다, 해당 방법론이 해결하려는 **핵심 문제와 연구 가정**을 명시한다.  
> 특히 sequence-level DPO의 preference aggregation 문제, interleaved visual-text reasoning에서의 modality shortcut, step-local modality dependency, modality ablation 이후의 uncertainty, 그리고 step-wise counterfactual 계산 비용을 수식으로 정의한다.
>
> **기준 reasoning 구조**  
> ILVR 계열의 interleaved reasoning:
>
> $$
> T_{k-1}
> \rightarrow
> V_k=(I_k^{\mathrm{helper}}, Z_k)
> \rightarrow
> T_k
> $$
>
> 여기서 최종적인 preference 및 uncertainty의 주요 관찰 단위는 다음 text reasoning block $T_k$이다.

---

## 1. 문제 설정

Multimodal long-form reasoning에서는 하나의 response가 여러 reasoning step으로 구성된다.  
특히 본 연구가 상정하는 interleaved reasoning trajectory는 text reasoning과 visual reasoning이 교차하는 구조를 가진다.

$$
\tau
=
(Q, I_0, T_0, V_1, T_1, V_2, T_2, \ldots, V_K, T_K)
$$

여기서 각 visual reasoning step은 개념적으로 다음과 같이 정의한다.

$$
V_k
=
(I_k^{\mathrm{helper}}, Z_k)
$$

즉,

$$
T_{k-1}
\rightarrow
I_k^{\mathrm{helper}}
\rightarrow
Z_k
\rightarrow
T_k
$$

라는 정보 흐름을 가정한다.

본 연구가 다루는 핵심 질문은 다음과 같다.

> **전체 response가 multimodal evidence에 의존하는가가 아니라, 각 reasoning transition에서 직전 text/visual reasoning state가 다음 reasoning step에 실제로 기여하고 있는가?**

---

# 2. Notation

본 문서 전체에서 다음 표기를 사용한다.

| 기호 | 정의 |
|---|---|
| $Q$ | 사용자 질문 또는 task instruction |
| $I_0$ | 문제의 initial input image |
| $I_k^{\mathrm{helper}}$ | $k$번째 visual reasoning step의 helper image |
| $Z_k$ | $I_k^{\mathrm{helper}}$와 이전 reasoning state를 바탕으로 형성되는 ILVR latent visual representation |
| $V_k$ | $k$번째 visual reasoning step, $V_k=(I_k^{\mathrm{helper}},Z_k)$ |
| $T_k$ | $k$번째 text reasoning block |
| $t_{k,j}$ | $T_k$의 $j$번째 token |
| $n_k$ | $T_k$의 token 수 |
| $P_{k-1}$ | $T_{k-1}$보다 이전의 clean reasoning history |
| $C_k^{+}$ | $T_k$ 생성 시 clean context |
| $C_k^{-V}$ | 직전 visual reasoning state의 기여를 제거한 counterfactual context |
| $C_k^{-T}$ | 직전 text reasoning state의 기여를 제거한 counterfactual context |
| $\pi_\theta$ | 현재 학습 중인 policy/model |
| $\pi_{\mathrm{ref}}$ | DPO에서 사용할 수 있는 frozen reference policy |
| $T_k^{+}$ | $k$번째 step의 preferred/correct text reasoning block |
| $T_{k,m}^{-}$ | modality $m$을 제거한 조건에서 얻은 rejected reasoning block |
| $s_\theta(T_k\mid C)$ | context $C$에서 $T_k$의 normalized log-likelihood score |
| $\Delta_k$ | $k$번째 reasoning step의 preference margin |
| $H[p]$ | predictive distribution $p$의 Shannon entropy |
| $K$ | 전체 text reasoning step 수 |

Text reasoning block의 normalized log-likelihood는 다음과 같이 정의한다.

$$
s_\theta(T_k\mid C)
=
\frac{1}{n_k}
\sum_{j=1}^{n_k}
\log
\pi_\theta
\left(
t_{k,j}
\mid
C,t_{k,<j}
\right)
$$

길이가 다른 reasoning block 간의 score scale 차이를 줄이기 위해 block 평균 log-likelihood를 기본 정의로 사용한다.

---

# 3. 이슈 1: Sequence-level DPO의 preference aggregation

## 3.1 Sequence-level preference margin

일반적인 sequence-level DPO에서는 전체 response에 대한 chosen/rejected preference를 하나의 scalar margin으로 집계한다.

Reasoning trajectory가 $K$개의 step으로 구성되고 각 step의 preference contribution을 $\Delta_k$라고 하면 전체 margin을 단순화하여

$$
\Delta_{\mathrm{seq}}
=
\sum_{k=1}^{K}\Delta_k
$$

로 볼 수 있다.

이에 대한 DPO loss는

$$
\mathcal L_{\mathrm{seq}}
=
-\log
\sigma
\left(
\beta
\Delta_{\mathrm{seq}}
\right)
$$

이다.

## 3.2 Global saturation

전체 margin에 대한 gradient magnitude는

$$
\left|
\frac{\partial \mathcal L_{\mathrm{seq}}}
{\partial \Delta_k}
\right|
=
\beta
\sigma
\left(
-\beta\Delta_{\mathrm{seq}}
\right)
$$

이다.

중요한 점은 이 multiplier가 **모든 step에 공통**이라는 것이다.

예를 들어

$$
\Delta_1=6,\quad
\Delta_2=3,\quad
\Delta_3=-1,\quad
\Delta_4=0
$$

이면

$$
\Delta_{\mathrm{seq}}=8
$$

이다.

전체 preference는 이미 충분히 강한 것으로 평가될 수 있지만,

$$
\Delta_3<0
$$

이므로 세 번째 reasoning step에서는 오히려 잘못된 preference가 남아 있다.

그럼에도

$$
\sigma(-8\beta)
$$

가 작아지면 $\Delta_3$에 전달되는 gradient까지 동시에 감소한다.

따라서 핵심 문제는 다음과 같이 정의한다.

> **일부 token 또는 reasoning step이 sequence-level preference margin을 먼저 크게 만들면, 아직 잘못 정렬된 다른 시점의 preference까지 충분한 gradient를 받지 못할 수 있다.**

이를 본 문서에서는 **sequence-level preference saturation** 또는 **cross-step preference interference**라고 부른다.

---

# 4. 이슈 2: Preference signal concentration과 multimodal shortcut

Multimodal reasoning에서는 위 문제가 단순한 optimization 문제를 넘어 modality shortcut으로 이어질 수 있다.

전체 trajectory가

$$
T_0
\rightarrow
V_1
\rightarrow
T_1
\rightarrow
V_2
\rightarrow
T_2
\rightarrow \cdots
$$

라고 하자.

전체 preference margin은 개념적으로

$$
\Delta_{\mathrm{seq}}
=
\Delta_{T_0}
+
\Delta_{V_1}
+
\Delta_{T_1}
+
\Delta_{V_2}
+
\Delta_{T_2}
+\cdots
$$

의 형태를 가진다.

하지만 text token이나 final answer는 visual grounding보다 상대적으로 쉬운 preference signal을 제공할 수 있다.

예를 들어,

$$
\Delta_{T_0}+\Delta_{T_K}\gg 0
$$

이지만 중간 visual reasoning과 관련된 step에서는

$$
\Delta_{V_k}\approx 0
$$

또는 잘못된 dependency가 남아 있을 수 있다.

이 경우 sequence-level objective는 이미 만족되지만, 모델은 중간 visual evidence를 실제로 사용하지 않아도 된다.

따라서 다음 두 명제는 동일하지 않다.

$$
\text{전체 response가 preferred}
$$

와

$$
\text{각 visual-text reasoning transition이 올바르게 grounded}
$$

는 서로 다른 요구사항이다.

본 연구가 다루는 문제는 후자이다.

---

# 5. 이슈 3: Interleaved reasoning에서 modality dependency의 위치가 불명확함

## 5.1 기본 reasoning unit

본 연구의 기본 transition은 다음과 같다.

$$
T_{k-1}
\rightarrow
V_k
\rightarrow
T_k
$$

그리고

$$
V_k=(I_k^{\mathrm{helper}},Z_k)
$$

로 정의한다.

여기서 $I_k^{\mathrm{helper}}$는 visual evidence의 source이고, $Z_k$는 해당 visual evidence가 reasoning state에 전달되는 latent representation이다.

본 연구에서는 $T_k$를 다음 reasoning step의 prediction 및 preference scoring unit으로 둔다.

---

## 5.2 Clean context

현재 step $T_k$ 이전의 clean context를

$$
C_k^{+}
=
(Q,I_0,P_{k-1},T_{k-1},Z_k)
$$

로 정의한다.

여기서 $P_{k-1}$은 $T_{k-1}$보다 이전의 모든 clean history이다.

---

## 5.3 Visual marginal dependency

직전 visual state $Z_k$의 marginal contribution을 보기 위한 context를

$$
C_k^{-V}
=
(Q,I_0,P_{k-1},T_{k-1},\operatorname{Ablate}(Z_k))
$$

로 정의한다.

그때 visual marginal effect의 일반적인 score는

$$
D_{k}^{V}
=
s_\theta(T_k\mid C_k^{+})
-
s_\theta(T_k\mid C_k^{-V})
$$

로 표현할 수 있다.

---

## 5.4 Text marginal dependency

직전 text state $T_{k-1}$의 marginal contribution을 보기 위한 context를

$$
C_k^{-T}
=
(Q,I_0,P_{k-1},\operatorname{Ablate}(T_{k-1}),Z_k)
$$

로 정의한다.

Text marginal effect는

$$
D_{k}^{T}
=
s_\theta(T_k\mid C_k^{+})
-
s_\theta(T_k\mid C_k^{-T})
$$

로 표현할 수 있다.

---

# 6. 이슈 4: Local marginal effect와 total causal effect의 구분

위 ablation은 **clean history를 보존한 상태에서 직전 component만 제거**한다.

따라서 본 방법이 측정하는 것은 total causal effect가 아니다.

예를 들어

$$
T_{k-1}
\rightarrow
Z_k
\rightarrow
T_k
$$

에서 $T_{k-1}$의 정보 일부가 이미 $Z_k$에 들어가 있을 수 있다.

그러므로

$$
C_k^{-T}
=
(Q,I_0,P_{k-1},\cancel T_{k-1},Z_k)
$$

에서도 $Z_k$를 통해 이전 text 정보가 일부 전달될 수 있다.

반대로 이전 visual information이 $T_{k-1}$에 이미 verbalize되어 있다면 $Z_k$를 제거해도

$$
I
\rightarrow
T_{k-1}
\rightarrow
T_k
$$

경로를 통해 visual information이 남을 수 있다.

따라서 본 연구의 primary quantity는 다음과 같다.

$$
\boxed{
\text{clean reasoning history가 고정되었을 때,
직전 reasoning component가 현재 step에 제공하는 marginal contribution}
}
$$

즉,

$$
D_k^{V}
$$

와

$$
D_k^{T}
$$

를 total modality dependence라고 해석하지 않는다.

이 구분은 연구 claim의 범위를 제한하면서 동시에 intervention의 의미를 명확히 한다.

---

# 7. 이슈 5: Visual reasoning에서 helper image와 latent representation의 역할

Visual reasoning step을

$$
V_k=(I_k^{\mathrm{helper}},Z_k)
$$

로 정의하는 이유는 visual modality가 source와 representation의 두 수준으로 존재하기 때문이다.

## 7.1 Source-level visual evidence

$$
I_k^{\mathrm{helper}}
$$

는 $k$번째 visual reasoning step이 참고해야 하는 visual evidence의 source이다.

## 7.2 Representation-level visual state

$$
Z_k
$$

는 해당 visual evidence와 이전 reasoning state가 model 내부에 반영된 latent representation이다.

따라서 visual ablation에는 최소 두 종류의 의미가 존재한다.

### Source-level intervention

$$
I_k^{\mathrm{helper}}
\rightarrow
\widetilde I_k^{\mathrm{helper}}
\rightarrow
\widetilde Z_k
\rightarrow
T_k
$$

이는 visual source 자체를 변경한다.

### Representation-level intervention

$$
I_k^{\mathrm{helper}}
\rightarrow
Z_k
\quad\text{but}\quad
T_k\not\leftarrow Z_k
$$

이는 clean visual state $Z_k$를 만든 후 현재 text reasoning step에서 그 접근만 차단한다.

두 intervention은 동일한 causal question을 측정하지 않는다.

본 연구의 현재 초안은 **step-local marginal dependency**를 우선하기 때문에 representation-level intervention을 primary setting으로 가정하고, source-level intervention은 보조 분석 대상으로 둔다.

---

# 8. 이슈 6: Ablation operator 자체가 새로운 confound를 만들 수 있음

$\operatorname{Ablate}(\cdot)$는 단순한 구현 세부사항이 아니다.

잘못된 ablation은 modality information을 제거하는 대신 새로운 artifact를 도입한다.

예를 들어 latent를 물리적으로 삭제하면 sequence position이 달라질 수 있다.

$$
Z_k\text{ deletion}
\Rightarrow
\text{position shift}
$$

그러면

$$
p(T_k\mid C_k^{+})
$$

와

$$
p(T_k\mid C_k^{-V})
$$

의 차이에 visual information뿐 아니라 positional perturbation까지 섞인다.

Zero vector replacement는

$$
Z_k\rightarrow 0
$$

와 같이 단순하지만, 모델이 학습 과정에서 보지 못한 out-of-distribution latent일 수 있다.

따라서 ablation operator는 다음 조건을 만족해야 한다.

1. 제거하려는 information path를 명확하게 차단할 것.
2. 가능하면 sequence length와 logical position을 유지할 것.
3. 제거된 modality 외의 context를 동일하게 유지할 것.
4. 모델이 ablation artifact 자체를 shortcut signal로 이용하지 못하게 할 것.
5. multiple operator 간 결과가 지나치게 달라지는지 검증할 수 있을 것.

현재 고려되는 operator 후보는 다음과 같다.

$$
\operatorname{Ablate}(Z_k)
\in
\{
\text{attention masking},
\text{null replacement},
\text{perturbation},
\text{dropout}
\}
$$

구체적인 선택은 방법론 명세에서 별도로 확정한다.

---

# 9. 이슈 7: Preference와 uncertainty는 다른 목표임

## 9.1 DPO가 보장하는 것

DPO는 chosen response와 rejected response의 상대적 preference를 학습한다.

Step $k$에서

$$
T_k^{+}\succ T_{k,m}^{-}
$$

를 학습한다고 하면, 핵심 목표는

$$
s_\theta(T_k^{+}\mid C)
>
s_\theta(T_{k,m}^{-}\mid C)
$$

이다.

즉 DPO가 답하는 질문은 다음과 같다.

> **어떤 reasoning continuation이 더 좋은가?**

---

## 9.2 DPO가 보장하지 않는 것

Modality가 제거되었을 때 preferred response의 probability가 감소했다고 해서 predictive distribution 전체가 불확실해지는 것은 아니다.

예를 들어

$$
p^{+}
=
(0.8,0.1,0.1)
$$

에서 modality 제거 후

$$
p^{-}
=
(0.1,0.8,0.1)
$$

이 되었다고 하자.

원래 정답의 probability는 크게 감소했지만,

$$
H(p^{-})
$$

는 여전히 낮다.

즉 모델은 정답 대신 다른 shortcut answer에 강한 confidence를 가질 수 있다.

---

# 10. 이슈 8: Modality 제거 후 predictive uncertainty

Ablated branch에서 모델이 충분히 불확실해져야 한다는 요구는 DPO와 별개의 calibration 문제이다.

Token $t_{k,j}$ 직전의 clean next-token distribution을

$$
p_{k,j}^{+}
=
\pi_\theta
\left(
\cdot
\mid
C_k^{+},t_{k,<j}
\right)
$$

라고 한다.

Modality $m\in\{V,T\}$를 제거한 distribution은

$$
p_{k,j}^{-m}
=
\pi_\theta
\left(
\cdot
\mid
C_k^{-m},t_{k,<j}
\right)
$$

로 정의한다.

Shannon entropy는

$$
H(p)
=
-\sum_{v\in\mathcal V}
p(v)\log p(v)
$$

이다.

여기서 $\mathcal V$는 language model vocabulary이다.

본 연구의 핵심 uncertainty hypothesis는 다음과 같다.

$$
H(p_{k,j}^{-m})
>
H(p_{k,j}^{+})
$$

즉 현재 reasoning step에 필요한 modality가 제거되면 predictive uncertainty가 증가해야 한다.

다만 다음과 같이 vocabulary-wide uniform distribution을 무조건 요구하는 것은 지나칠 수 있다.

$$
p_{k,j}^{-m}
\rightarrow
U_{\mathcal V}
$$

왜냐하면 function word나 문법 token은 modality가 없어도 충분히 예측 가능할 수 있기 때문이다.

따라서 중요한 연구 문제는 단순한 entropy maximization이 아니라 다음과 같다.

> **modality 제거가 clean prediction에 비해 충분한 uncertainty increase를 만들도록 하되, language modeling 자체를 붕괴시키지 않는가?**

이 때문에 absolute maximum entropy와 clean-relative entropy margin을 구분해야 한다.

---

# 11. 이슈 9: Step-wise DPO와 entropy regularization의 역할 분리

현재 방법론에서는 두 objective를 서로 다른 목적으로 사용한다.

## Preference objective

$$
\mathcal L_{\mathrm{StepDPO}}
$$

는 다음을 담당한다.

$$
T_k^{+}
\succ
T_{k,m}^{-}
$$

즉 올바른 reasoning continuation을 잘못된 continuation보다 선호하게 만든다.

## Uncertainty objective

$$
\mathcal L_{\mathrm{Entropy}}
$$

는 다음을 담당한다.

$$
H(p_{k,j}^{-m})
>
H(p_{k,j}^{+})
$$

즉 필요한 modality가 제거되었을 때 모델의 confidence를 낮춘다.

두 objective는 상보적이다.

$$
\boxed{
\text{Preference: 무엇을 선택해야 하는가}
}
$$

$$
\boxed{
\text{Entropy: evidence가 없을 때 얼마나 확신하지 말아야 하는가}
}
$$

따라서 둘을 하나의 rejected likelihood로 동일시하지 않는다.

---

# 12. 이슈 10: Step-wise supervision의 negative construction

Step-wise genuine DPO를 사용하려면 각 $T_k$에 대한 실제 rejected reasoning block이 필요하다.

Modality $m$을 제거한 상태에서 rejected block을

$$
T_{k,m}^{-}
\sim
\pi
\left(
\cdot
\mid
C_k^{-m}
\right)
$$

로 만들 수 있다.

여기서 발생하는 문제는 다음과 같다.

1. 생성된 negative가 실제로 incorrect한가?
2. clean positive와 거의 동일한 negative는 학습 signal이 있는가?
3. 너무 쉬운 negative만 생성되면 DPO가 trivial해지지 않는가?
4. online generation을 사용하면 rollout compute가 지나치게 증가하지 않는가?
5. frozen checkpoint가 만든 negative가 현재 policy에 비해 너무 stale해지지 않는가?

따라서 negative dataset은 단순한 sampling 결과가 아니라 **valid preference pair**인지 검증되어야 한다.

본 연구에서는 이 문제를 preference data quality 문제로 별도 취급한다.

---

# 13. 이슈 11: Reference model과 interleaved latent trajectory

Standard DPO는 일반적으로 frozen reference policy를 사용한다.

$$
\pi_{\mathrm{ref}}
$$

그러나 ILVR처럼 intermediate latent state $Z_k$가 존재하면 policy와 reference가 서로 다른 latent trajectory를 만들 수 있다.

현재 policy가 생성한 latent를

$$
Z_k^{\theta}
$$

reference가 생성한 latent를

$$
Z_k^{\mathrm{ref}}
$$

라고 하면,

$$
s_\theta(T_k\mid Z_k^\theta)
-
s_{\mathrm{ref}}(T_k\mid Z_k^{\mathrm{ref}})
$$

에는 단순한 policy preference 차이뿐 아니라 **latent context 자체의 차이**가 포함될 수 있다.

따라서 reference model을 사용할 경우 다음 중 무엇을 비교하는지 명시해야 한다.

### Same-context comparison

정책과 reference에 동일한 clean latent/context를 제공한다.

$$
Z_k^{\theta}
=
Z_k^{\mathrm{ref}}
=
Z_k^{\mathrm{shared}}
$$

### Independent-context comparison

각 모델이 자신의 latent trajectory를 사용한다.

$$
Z_k^{\theta}
\neq
Z_k^{\mathrm{ref}}
$$

두 방식은 서로 다른 objective를 정의한다.

따라서 reference selection뿐 아니라 **reference context alignment**가 DPO formulation의 일부이다.

현재 ours1은 위의 independent-context comparison을 구현한다. Frozen
Stage 2A가 자체 latent로 clean chosen/rejected score를 계산하고 offline
scalar artifact에 저장하며, Stage 2B가 현재 policy latent를 reference에
주입하지 않는다. 이번 shared-prefix 최적화는 이 conditioning을 유지한다.
Cache 재사용은 reference의 전체 입력과 수치 설정이 고정된 경우에만
유효하다. 기존 manifest가 외부 image byte 전체를 인증하지 않는 한계를
별도로 audit하고, 누락·불일치 cache는 hit나 0점으로 처리하지 않는다.

---

# 14. 이슈 12: Step-local counterfactual의 계산 비용

## 14.1 Naive computation

각 text reasoning step $T_k$에서 clean, V-off, T-off 조건을 독립적으로 계산하면 대략

$$
1+2K
$$

개의 logical branch가 필요하다.

각 branch에서 전체 prefix를 다시 계산하면 계산량은 크게 증가한다.

전체 trajectory token 길이를 $L$이라 하면 단순 구현은 개념적으로

$$
\mathcal O(KL)
$$

수준의 반복 prefix 계산을 발생시킬 수 있다.

---

## 14.2 실제 중복되는 부분

그러나 각 counterfactual branch는 분기 전까지 동일한 clean prefix를 공유한다.

즉,

$$
P_{k-1}
$$

까지의 computation은 V-off와 T-off branch에서 동일하다.

구조적으로는 다음과 같은 tree이다.

```text
clean trajectory
T0 ── Z1 ── T1 ── Z2 ── T2 ── ... ── TK
            │             │
            ├─ V-off      ├─ V-off
            └─ T-off      └─ T-off
```

따라서 핵심 계산 문제는

> **각 step-local intervention의 독립성을 유지하면서 clean prefix computation을 얼마나 공유할 수 있는가?**

이다.

---

## 14.3 Forward call 수와 FLOPs는 다름

승인된 shared-prefix 확장의 clean/V/T는 세 논리 그룹이며, 실제 forward
세 번이라는 의미가 아니다. 기존 latent rollout 내부 순차 호출을 보존하고
suffix-only projection/MLP 처리량, chunk 수, backward checkpoint 재계산과
reference online 호출/cache 소비를 따로 계측해야 한다. 새 경로의 측정값은
과거 packed benchmark 값으로 대신할 수 없다.

Branch를 batch 또는 packed sequence로 묶어 API 수준의 forward call을 1회로 줄였다고 해서 실제 계산량이 자동으로 줄어드는 것은 아니다.

다음을 구분해야 한다.

$$
\text{number of forward calls}
$$

과

$$
\text{actual processed tokens / attention edges / FLOPs}
$$

는 동일하지 않다.

따라서 효율성 주장은 반드시 다음을 포함해야 한다.

- wall-clock throughput
- peak GPU memory
- processed token 수
- attention computation
- reasoning step 수 $K$에 따른 scaling

---

# 15. 이슈 13: Clean prefix gradient와 branch sharing

Inference에서의 KV-cache 공유와 training에서의 prefix sharing은 동일하지 않다.

만약 clean prefix의 hidden state 또는 K/V를 detach하여 branch에서 재사용하면,

$$
\mathcal L_{\mathrm{branch}}
\not\rightarrow
P_{k-1}
$$

가 되어 branch loss의 gradient가 clean prefix를 생성한 parameter로 전달되지 않는다.

반면 exact end-to-end training에서는

$$
\mathcal L_{\mathrm{branch}}
\rightarrow
P_{k-1}
\rightarrow
\theta
$$

가 유지되어야 할 수 있다.

따라서 계산 최적화는 다음 조건을 만족해야 한다.

$$
\boxed{
\text{prefix computation sharing}
+
\text{correct branch isolation}
+
\text{desired gradient preservation}
}
$$

이 세 조건을 동시에 만족해야 한다.

---

# 16. 이슈 14: ILVR Stage 1과 Stage 2의 distribution 차이

ILVR에서는 visual latent $Z_k$의 의미가 Stage에 따라 다르다.

## Stage 1

Helper image 기반 teacher supervision을 사용하여 latent semantics를 bootstrap한다.

개념적으로

$$
I_k^{\mathrm{helper}}
\xrightarrow{\mathrm{teacher}}
Z_k^{\mathrm{target}}
$$

의 강한 supervision이 존재한다.

## Stage 2

Teacher alignment를 완화하고 self-generated latent를 사용하여 downstream reasoning을 학습한다.

$$
T_{k-1}
\rightarrow
Z_k^{\theta}
\rightarrow
T_k
$$

따라서 modality dependency를 실제 inference-time behavior에 가깝게 학습하려면, teacher-forced latent와 self-generated latent를 동일한 상태로 취급해서는 안 된다.

이것이 preference/entropy alignment를 Stage 2 계열에서 다루어야 하는 주요 이유이다.

---

# 17. 핵심 연구 가설

현재까지의 논의를 다음 네 가설로 요약한다.

## H1. Sequence-level preference saturation

$$
\boxed{
\text{전체 sequence margin의 포화는
아직 잘못 정렬된 reasoning step의 gradient를 약화시킬 수 있다.}
}
$$

## H2. Step-local modality alignment

$$
\boxed{
\text{각 }T_k\text{에 대한 preference를 독립적으로 학습하면
중간 reasoning transition의 modality grounding을 직접 감독할 수 있다.}
}
$$

## H3. Modality-removal uncertainty

$$
\boxed{
\text{필요한 text/visual reasoning state가 제거되면
clean condition보다 predictive entropy가 증가해야 한다.}
}
$$

## H4. Shared-prefix efficiency

$$
\boxed{
\text{step-local counterfactual branch는 동일한 clean prefix를 공유하므로
naive }O(K)\text{ 반복 계산의 상당 부분을 제거할 수 있다.}
}
$$

---

# 18. 현재 연구에서 주장하지 않는 것

과도한 claim을 피하기 위해 다음은 현재 primary claim에서 제외한다.

1. $D_k^V$ 또는 $D_k^T$가 modality의 total causal effect라는 주장.
2. $Z_k$가 순수한 visual information만 담는다는 주장.
3. T-off가 모든 과거 textual information을 제거한다는 주장.
4. V-off가 모든 과거 visual information을 제거한다는 주장.
5. entropy 증가가 곧 reasoning correctness를 보장한다는 주장.
6. tree packing 자체가 새로운 attention algorithm이라는 주장.
7. helper-image perturbation과 latent masking이 동일한 intervention이라는 주장.

현재 primary claim은 더 제한적이다.

> **Interleaved reasoning의 각 text transition에서 직전 text/visual reasoning component의 marginal contribution을 preference 및 uncertainty 관점에서 직접 감독한다.**

---

# 19. 방법론으로부터 요구되는 설계 조건

위 이슈들로부터 향후 구현은 최소 다음 요구사항을 만족해야 한다.

### R1. Step identity 보존

각 $T_k$, $Z_k$, $T_{k-1}$의 token/latent span이 preprocessing 이후에도 식별 가능해야 한다.

### R2. Local intervention

$C_k^{-V}$와 $C_k^{-T}$는 그보다 이전의 clean history $P_{k-1}$을 유지해야 한다.

### R3. Stable ablation

Ablation은 position shift, token count 변화, 명백한 OOD artifact를 가능한 한 최소화해야 한다.

### R4. Separate objectives

Preference loss와 entropy loss는 서로 다른 목적 함수로 계산 가능해야 한다.

### R5. Valid rejected steps

DPO negative는 modality-off 조건에서 얻은 의미 있는 hard negative여야 한다.

### R6. Exact reference baseline

효율화 전에는 독립 forward 방식으로 loss와 gradient의 기준값을 확보해야 한다.

### R7. Packed equivalence

최적화된 shared-prefix/tree-packed 구현은 naive 구현과 numerical loss 및 gradient가 일치하는지 검증해야 한다.

---

# 20. 문제 정의의 최종 요약

현재 연구가 해결하려는 문제는 다음 한 문장으로 요약할 수 있다.

> **Long-form interleaved multimodal reasoning에서 sequence-level preference optimization은 일부 쉬운 token 또는 reasoning step의 preference signal에 의해 먼저 포화될 수 있고, 그 결과 중간 단계의 modality dependency와 modality 제거 시의 uncertainty가 충분히 학습되지 않을 수 있다.**

이를 보다 구조적으로 표현하면 다음과 같다.

$$
\boxed{
\begin{aligned}
&\text{Problem A:}
&&
\text{sequence-level preference aggregation}
\rightarrow
\text{cross-step saturation}
\\[4pt]
&\text{Problem B:}
&&
\text{text/final-answer shortcut}
\rightarrow
\text{weak intermediate visual grounding}
\\[4pt]
&\text{Problem C:}
&&
\text{modality ablation}
\not\Rightarrow
\text{uncertainty increase}
\\[4pt]
&\text{Problem D:}
&&
\text{step-local intervention}
\rightarrow
\text{naive }O(K)\text{ repeated computation}
\end{aligned}
}
$$

따라서 이후 방법론은 다음 네 요구를 동시에 만족해야 한다.

$$
\boxed{
\text{step-wise preference}
+
\text{local modality intervention}
+
\text{ablated-branch uncertainty calibration}
+
\text{shared-prefix efficient computation}
}
$$

이 네 항목이 현재 확립된 방법론을 지지하는 핵심 문제 정의이다.
