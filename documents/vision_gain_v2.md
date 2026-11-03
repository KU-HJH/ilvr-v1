# Stage 1 Vision Gain 통합 설계 명세

**적용 대상:** ILVR Stage 1의 기존 단일 학습 루프  
**문서 상태:** 구현을 위한 설계 제안. 모델 학습 효과는 미검증  
**통합 기준:** `stage1_vision_gain_spec.md`의 보조 방향 projection 설계

## 1. 목적과 범위

**기존 Stage 1을 유지하면서, teacher-selected visual evidence가 블록 뒤 GT 텍스트 예측에 기여하도록 gain 보조 업데이트를 추가한다.** 보조 방향에서 masked 점수를 낮추는 성분을 제거한다.

Teacher-selected feature를 latent 입력과 alignment target으로 사용하는 경로, text teacher forcing, 원본 이미지 입력, teacher의 feature 선택·EMA, 모델 구조, 기존 optimizer, Stage 2는 변경하지 않는다.

Calibration, 고정 reference, 추가 warm-up, self-generated latent 학습 전환, bi-level은 도입하지 않는다. 전체 optimizer update를 제약하는 이전 안도 사용하지 않는다. **Projection 대상은 gain 보조 방향 하나뿐이다.**

본 문서는 앞선 설계안들을 대신하는 단일 기준 문서다. 기존 코드의 동작을 기록한 `vision_gain_v1.md`는 구현 비교 자료이며, 본 설계가 이미 구현되어 있다는 근거가 아니다.

## 2. 수학적 notation

| 기호 | 정의 |
|---|---|
| $\theta\in\mathbb R^P$ | 기존 Stage 1에서 학습 가능한 student 파라미터를 펼친 벡터 |
| $x,y$ | 원본 이미지·질문과 GT 텍스트 trajectory |
| $Z^T$ | Teacher-selected visual features. 동일 비교의 두 경로에서 같으며 student gradient에 대해서는 고정 |
| $b$ | 샘플 식별자를 포함한 visual latent 블록 식별자 |
| $l_b,r_b$ | 원본 sequence에서 블록의 latent start/end marker 위치 |
| $F$ | 기존 Stage 1의 정상 teacher-forced forward |
| $M_b$ | 블록 $b$의 latent body를 읽는 attention만 차단한 forward |
| $W_b,T_b=\lvert W_b\rvert$ | 블록 바로 다음 text item의 유효 GT target 위치 집합과 크기 |
| $\mathcal B,B=\lvert\mathcal B\rvert$ | 한 optimizer update의 전체 rank·누적 microbatch에 포함된 scoring-valid 블록 집합과 크기 |
| $s_b^F,s_b^M$ | 각 경로의 블록별 평균 GT log-probability |
| $S^F,S^M$ | 각 경로의 전역 블록 평균 점수 |
| $G=S^F-S^M$ | 현재 optimizer update의 평균 vision gain |
| $\mathcal L_{\mathrm{S1}}$ | 기존 Stage 1의 CE와 alignment loss |
| $\lambda_{\mathrm{align}}$ | 기존 Stage 1의 alignment 가중치. 변경하지 않음 |
| $g=\nabla_\theta G$ | Gain을 높이는 raw 보조 방향 |
| $h=\nabla_\theta S^M$ | Masked 점수를 높이는 방향 |
| $u$ | Masked 점수 악화 성분을 제거한 보조 방향 |
| $\Delta_{\mathrm{S1}}$ | 기존 Stage 1 loss만으로 optimizer가 계산하는 실제 parameter displacement |
| $\alpha_{\mathrm{vis}}\ge0$ | 보조 parameter update의 크기. 검증된 최적값은 없음 |

이하 $g,h,u$의 내적과 norm은 동일한 학습 가능 파라미터 공간에서 계산한다. $Z^T$를 고정한다는 것은 student 역전파의 대상이 아니라는 뜻이며, 별도의 frozen reference teacher를 추가한다는 뜻이 아니다.

## 3. Forward와 GT scoring

### 3.1. 정상 경로와 기존 loss

기존 Stage 1과 동일하게 text와 latent 입력을 teacher-force한다. Teacher feature는 입력과 alignment target으로 사용한다. Self-generated latent rollout은 추가하지 않는다.

$$
\mathcal L_{\mathrm{S1}}
=
\mathcal L_{\mathrm{CE}}^F
+
\lambda_{\mathrm{align}}\mathcal L_{\mathrm{align}}^F.
$$

Text CE의 supervised label 범위, alignment 정의·가중치, 기존 loss의 정규화는 유지한다. CE를 아래 $W_b$에만 제한하지 않는다. Teacher target에는 student gradient를 전달하지 않는다.

### 3.2. 차단 경로

각 $M_b$는 $F$와 같은 원본 입력, teacher feature 값, GT prefix, token ID, marker와 절대 위치를 사용한다. 블록별로 독립적으로 차단하며 다른 블록의 차단을 누적하지 않는다.

개입은 `latent_body`로 고정한다. Query 위치를 $q$, key 위치를 $k$라고 할 때, 기존 causal·padding mask에 아래 규칙을 추가한다.

$$
q\ge r_b,\qquad l_b<k<r_b
\quad\Longrightarrow\quad
A_{qk}=-\infty.
$$

모든 관련 attention layer에 동일하게 적용한다. **End marker와 이후 hidden state는 masked 조건에서 다시 계산한다.** 이미 latent를 읽은 clean end-marker state나 downstream KV cache를 재사용하지 않는다.

원본 이미지, marker, token 위치는 제거하지 않는다. 따라서 이 개입은 전체 vision-off가 아니라 **선택한 latent body의 소비 경로에 대한 ablation**이다. 가능한 한 두 경로의 확률적 연산에 같은 난수 실현을 사용하여 mask 외 차이를 통제한다.

Masked 경로에도 student gradient를 계산한다. Masked score에 `detach()`나 `no_grad()`를 사용하지 않는다.

### 3.3. GT window

$W_b$는 **블록 바로 뒤에 인접한 text item 전체의 유효 supervised text target**이다. Padding, label이 `-100`인 위치, non-text target은 제외한다. 그다음 text item이나 다음 latent 블록까지 확장하지 않는다.

$T_b=0$이면 해당 블록을 scoring에서 제외한다. Pre window는 요구하지 않으며, pre가 비었다는 이유로 블록을 제외하지 않는다. 별도의 semantic·decision-span annotation은 요구하지 않는다.

따라서 아래 점수는 **local GT text prediction의 지표**다. 모든 토큰이 시각 판단을 나타내거나, 이 점수가 최종 추론 정확도와 같다고 가정하지 않는다.

### 3.4. 점수와 reduction

$$
s_b^F
=
\frac1{T_b}\sum_{i\in W_b}
\log p_\theta^F(y_i\mid x,y_{<i},Z^T),
$$

$$
s_b^M
=
\frac1{T_b}\sum_{i\in W_b}
\log p_\theta^{M_b}(y_i\mid x,y_{<i},Z^T).
$$

Target 위치 $i$는 predictor 위치 $i-1$의 logit에서 점수화한다. FP32로 변환한 logits의 **전체 vocabulary log-softmax**에서 GT ID를 gather한다. Text vocabulary만으로 재정규화하지 않는다.

$B>0$일 때,

$$
S^F=\frac1B\sum_{b\in\mathcal B}s_b^F,
\qquad
S^M=\frac1B\sum_{b\in\mathcal B}s_b^M,
\qquad
G=S^F-S^M.
$$

블록 내부에서는 토큰 평균, 블록 사이에서는 전역 블록 평균을 사용한다. 두 경로는 같은 블록 집합과 같은 denominator를 사용한다.

## 4. Gain loss와 보조 방향

Raw gain loss는 다음과 같다. 고정 margin이나 softplus는 사용하지 않는다.

$$
\mathcal L_{\mathrm{gain}}=-G.
$$

**동일한 업데이트 전 파라미터**에서 두 경로 모두 미분한다.

$$
g=-\nabla_\theta\mathcal L_{\mathrm{gain}}
=\nabla_\theta S^F-\nabla_\theta S^M,
\qquad
h=\nabla_\theta S^M.
$$

$h^\top g<0$이면 raw gain 방향은 masked 점수를 낮추는 일차 성분을 포함한다. 실제 보조 방향은 다음 projection으로 정한다.

$$
\boxed{
u=
\begin{cases}
g, & h=0,\\[2mm]
g-\dfrac{\min(0,h^\top g)}{\|h\|_2^2}h, & h\ne0.
\end{cases}
}
$$

이는 아래 문제의 닫힌형 해다.

$$
u=
\arg\min_d\frac12\|d-g\|_2^2
\quad\text{subject to}\quad h^\top d\ge0.
$$

즉, **gain을 높이는 raw 방향을 최대한 유지하면서 masked 점수를 낮추는 성분만 제거한다.** Clean·masked가 같은 점수 gradient를 가지면 $g=0$이므로 새로운 보조 신호가 생기지 않는다.

## 5. 실제 parameter update

$$
\boxed{
\theta_{\mathrm{new}}
=
\theta+\Delta_{\mathrm{S1}}+\alpha_{\mathrm{vis}}u.
}
$$

$\Delta_{\mathrm{S1}}$는 기존 optimizer가 $\mathcal L_{\mathrm{S1}}$의 gradient만으로 계산한다. 기존 momentum, preconditioning, weight decay 등은 이 기본 업데이트 안에서 유지한다. 기존 optimizer state도 기존 Stage 1 gradient로만 갱신한다.

**$\alpha_{\mathrm{vis}}u$는 gradient buffer에 합쳐 AdamW 등을 다시 통과시키지 않고, parameter displacement로 더한다.** $u$를 좌표별로 재스케일하면 §7의 성질이 일반적으로 유지되지 않는다.

전체 optimizer displacement를 projection하는 것이 아니다. $\Delta_{\mathrm{S1}}$는 그대로 두고 보조 방향 $u$만 보정한다. 따라서 이 알고리즘을 단순히 $\mathcal L_{\mathrm{S1}}+\lambda\mathcal L_{\mathrm{gain}}$를 기존 optimizer로 최적화하는 방식으로 대체하면 안 된다.

## 6. 한 optimizer update의 실행 계약

다음은 한 학습 루프 안의 계산 순서이며, 별도 학습 phase가 아니다.

1. **Forward:** 동일한 업데이트 전 파라미터에서 정상 Stage 1 forward와 유효 블록별 독립 masked forward를 계산한다. Teacher feature는 비교 중 동일하게 유지한다.
2. **Gradient 계산·집계:** 기존 Stage 1 gradient와 $g,h$를 구한다. 전체 rank·누적 microbatch를 집계하여 §3.4의 전역 평균에 맞춘다. Projection 전에 AMP loss scaling을 제거한다.
3. **Projection:** 전역 $g,h$로 §4의 $u$를 한 번 계산한다. Rank·microbatch별 projection 결과를 평균하는 방식은 사용하지 않는다.
4. **Parameter 갱신:** 기존 Stage 1 optimizer의 이동량 $\Delta_{\mathrm{S1}}$와 보조 이동량 $\alpha_{\mathrm{vis}}u$를 §5대로 적용한다. 관련 gradient는 모두 갱신 전 파라미터에서 계산되어 있어야 한다.
5. **Teacher 유지:** 기존 EMA teacher 갱신 규칙을 유지한다. 별도 teacher 학습이나 calibration은 수행하지 않는다.

$B=0$ 또는 $\alpha_{\mathrm{vis}}=0$이면 masked 계산과 보조 업데이트를 생략하고 기존 Stage 1만 수행한다. $h=0$이면 $u=g$다. $h\ne0$이지만 분모가 수치적으로 불안정할 만큼 작으면 해당 보조 업데이트를 생략한다. 분모에 임의의 epsilon을 더하고 정확한 projection이라고 취급하지 않는다.

추가 비용은 masked backward와 별도의 score-gradient buffer·집계다. 실제 ILVR 실행시간과 메모리 사용량은 측정되지 않았다.

## 7. 수학적 성질과 보장의 한계

### 7.1. 보조 방향에서 성립하는 것

정확한 산술에서 §4의 projection은 다음을 만족한다.

$$
h^\top u\ge0,
\qquad
g^\top u=\|u\|_2^2\ge0.
$$

또한 $\nabla_\theta S^F=g+h$이므로,

$$
(\nabla_\theta S^F)^\top u
=
\|u\|_2^2+h^\top u
\ge0.
$$

따라서 현재 파라미터에서 **보조 방향 $u$만의 현재 batch 평균에 대한 방향미분**은 masked 점수, gain, clean 점수 모두 음수가 아니다. $u\ne0$이면 gain의 방향미분은 양수다.

### 7.2. 이 설계가 보장하지 않는 것

위 성질은 유한한 step 이후의 실제 점수 보존을 보장하지 않는다. 기본 업데이트 $\Delta_{\mathrm{S1}}$까지 더한 전체 업데이트, 개별 블록, 다른 batch나 평가셋에 대해서도 단조 개선을 보장하지 않는다. CE와 alignment loss의 단조 감소도 추가로 주장하지 않는다.

Projection 후 $u=0$이 될 수 있다. Gain gradient가 0이거나, raw 방향에서 masked 악화 성분을 제거한 뒤 남는 방향이 없으면 보조 이동을 하지 않는다.

학습 대상은 Stage 1의 **teacher-forced evidence 활용**이다. Self-generated latent의 유용성, 최종 추론 정확도, Stage 2 이후 효과 유지는 별도 검증 대상이다. Attention ablation gain만으로 latent의 시각적 의미 활용 전체를 입증하지 않는다.

## 8. 필수 확인 항목

### 8.1. 구현 검사

검사는 다음 조건의 충족 여부를 확인하는 요구사항이며, 이 문서가 테스트 실행 결과를 제시하는 것은 아니다.

| 검사 대상 | 확인 조건 |
|---|---|
| Window와 scoring | 바로 다음 text item만 사용하고, 두 경로의 GT·predictor 좌표·블록 집합이 같음 |
| Masked 계산 | End marker 이후를 재계산하며, masked student gradient가 존재함 |
| Projection | 수치 허용오차 내 $h^\top u\ge0$, $g^\top u=\lVert u\rVert_2^2$, $(g+h)^\top u\ge0$ |
| Update 적용 | 실제 parameter delta가 $\Delta_{\mathrm{S1}}+\alpha_{\mathrm{vis}}u$와 일치하며, 보조 방향을 optimizer에 재입력하지 않음 |
| 경계 조건 | $B=0$ 또는 $\alpha_{\mathrm{vis}}=0$에서 기존 Stage 1로 복귀함 |
| 전역 집계 | Rank·microbatch 집계 후 한 번 projection하며, 두 경로의 정규화가 같음 |

### 8.2. 학습·평가 지표

동일한 window와 ablation으로 $S^F,S^M,G$를 각각 기록한다. Projection이 실제 적용된 비율과 $u=0$인 비율도 기록한다. **Gain 증가만으로 성공으로 판정하지 않는다.** 실제 clean 또는 masked 점수가 악화되면 방향미분의 성질과 별개로 그 결과를 보고한다.

비교 대상은 동일한 학습 조건의 수정 전 Stage 1이다. 이 비교를 위해 학습 중 고정 reference를 추가하지 않는다.

최종 목적은 self-generated latent를 사용하는 추론 정확도와 latent 개입 결과로 평가한다. Stage 2는 수정하지 않으며, 이후 Stage 2 학습까지 수행한 경우에는 그 최종 모델을 별도로 평가한다.

## 9. 기존 v1과의 차이 및 근거

첨부 `vision_gain_v1.md`의 loss는 다음과 같다.

$$
\ell_b^{\mathrm{v1}}
=
\operatorname{softplus}\bigl(m-s_b^F+\operatorname{sg}(s_b^M)\bigr),
$$

$$
\nabla_\theta\ell_b^{\mathrm{v1}}
=
\sigma(m-s_b^F+s_b^M)\,\nabla_\theta(-s_b^F).
$$

이는 clean NLL gradient의 재가중이며, masked score의 stop-gradient는 실제 점수 보존을 뜻하지 않는다. 본 문서는 이 gradient 구조를 다음처럼 변경한다.

| 항목 | 첨부 v1 | 본 명세 |
|---|---|---|
| 평가 window | 기본 `post_window=1`; vision-only에도 pre 필요 | 바로 다음 text item의 유효 GT 전체; pre 불필요 |
| Gain objective | Stop-gradient masked score를 사용하는 softplus | 양쪽을 미분하는 $-G$ |
| 보조 갱신 | Scalar loss로 합산 | $g$의 projection 후 보조 parameter displacement로 추가 |
| Masked 악화 대응 | 직접적인 masked 하강 gradient만 차단 | 보조 방향의 masked 하강 성분 제거 |
| Scoring·ablation | FP32 전체 vocabulary, `latent_body` 사용 가능 | 같은 scoring과 `latent_body` 사용 |

구현 비교 근거는 사용자 첨부 `vision_gain_v1.md`의 §2–7이며, 통합 설계의 직접 근거는 `stage1_vision_gain_spec.md`다. 기존 Stage 1 구성은 후자의 적용 전제를 유지했다. 이번 문서 통합에서 외부 논문이나 실제 저장소 코드를 추가 검증하지 않았다.

**검증 상태:** 본 문서는 기존 설계의 통합본이다. 이번 작업에서 실제 ILVR 코드 수정, 통합 테스트, 학습 또는 성능 평가를 수행하지 않았다. 원문에 기재된 과거 합성 벡터 검사에 대한 서술도 이번 작업에서 독립적으로 확인하지 않았으며, 모델 효과의 근거로 사용하지 않는다.
