# DIL-BCE-ORIG + Top-k Entropy — 구현 변경 명세 v1

작성일: 2026-10-09  
상태: **구현 제안 / 미구현 / 저장소·GPU 테스트 미실행**  
공개 alias: **`dil-bce-orig`**  
내부 canonical loss type: **`dil_bce_orig`**  
기준 문서: `dil-bce_current.md` — 「DIL-BCE v1 — 현재 구현 명세」, 기준일 2026-10-07.

## 0. 결정 요약과 문서의 지위

기존 `dil_bce_rf`, `dil_bce_rf_topk`를 수정하여 의미를 바꾸지 않는다. 새 alias를 추가하여 **clean의 chosen/reference DIL-BCE**와 **V-off의 top-k conditional entropy regularization**을 분리한다.

\[
\boxed{
L_{\mathrm{total}}
=\lambda_{\mathrm{CE}}L_{\mathrm{clean\ CE}}
+\lambda_P\bigl(L_{\mathrm{chosen}}+L_{\mathrm{reference}}\bigr)
+\lambda_U L_{\mathrm{topk\ entropy}}
+\lambda_M L_{\mathrm{mass}}.
}
\]

`lambda_topk_entropy=0`이면 새 preference loss는 chosen/reference DIL-BCE만 계산한다. 양수이면 동일 alias에서 V-off entropy를 추가한다. 여기서 “DIL-BCE만”은 preference 부분에 관한 말이며, clean CE의 유지 여부는 `lambda_ce`가 결정한다.

이 문서는 기준 명세의 1–11절 구조에 대응하는 **변경 명세**다. 기존 문서와 원본 파일은 보존한다. 이하 “기존”은 첨부 명세에 적힌 상태, “신규”와 명령형 규칙은 이번에 제안하는 구현 계약을 뜻한다. 소스 파일명과 함수명 중 기존으로 명시한 것은 첨부 명세에 등장하는 이름이다. 실제 저장소와 그 문서가 링크한 `2026-10-07-dil-bce-reference-design.md`는 제공되지 않아 내용을 확인하지 못했다. 해당 설계를 이미 구현한 것으로 전제하지 않는다.

### 0.1 반드시 구분할 세 항

| 항 | 입력 문맥 | 분포/표본 | 역할 |
|---|---|---|---|
| Chosen BCE | clean | 데이터의 GT token | chosen/reference 밀도비 추정의 positive 부분 |
| Reference BCE | 같은 clean 문맥 | 고정 reference의 full-vocabulary 분포 | 밀도비 추정의 reference expectation |
| Top-k entropy | 기존 시각적 개입의 V-off | 선택된 vocabulary 후보의 조건부 분포 | 후보 집합 내부의 균등화 |

**Reference 항을 V-off GT, V-off top-k, 또는 GT 제외 vocabulary로 대체하지 않는다.** Entropy도 reference 항을 대체하지 않는다. 별도 rejected 문장, reward model, PPO rollout은 추가하지 않는다.

### 0.2 `orig`의 정확한 의미

신규 v1은 원 DIL의 **chosen/reference BCE 밀도비 구조를 token 조건부 분포에 적용**한다. 원 논문의 응답 단위 목적함수를 그대로 재현한다는 뜻은 아니다. teacher-forced 문맥에서 각 token을 행동으로 삼아 reference expectation을 vocabulary 전체 합으로 정확히 계산한다.

기존 “token log 확률 평균 → step softplus”를 그대로 사용하면 이 token-level 이론형과 달라진다. 그러므로 새 alias만 **token별 BCE → step 안 token 평균 → step 평균**으로 정의한다. 기존 alias의 reduction은 변경하지 않는다.

## 1. 적용 범위와 alias 추가 — 기존 1절 수정 방향

### 1.1 Selector 계약

| 외부 selector | 내부 값 | 동작 |
|---|---|---|
| `dil_bce_rf` | 기존 값 유지 | 기존 reference-free GT clean/V-off BCE |
| `dil_bce_rf_topk` | 기존 값 유지 | 기존 GT union / `topk_all` 경로 |
| `dil-bce-orig` | `dil_bce_orig` | 신규 chosen/reference BCE + 선택적인 top-k entropy |
| `dil_bce_orig` | `dil_bce_orig` | 위 alias의 underscore 동의어 |

전역적으로 모든 문자열의 `-`를 `_`로 바꾸지 말고 명시적 alias map을 추가한다. 신규 alias를 생략한 전체 Stage 2B 기본 loss는 기존대로 유지한다. `dil_bce`라는 별도 selector를 임의로 신설하지 않는다.

### 1.2 유지하는 기능

기존 clean-prefix, predictor 위치 정렬, target span, reasoning-step 동일 가중 평균, clean CE 범위, visual-only 개입, `attention_mask` 및 허용된 `image_embedding_replacement` 분기를 재사용한다. 새 alias를 만들기 위해 기존 이미지 선택, latent 개입, branch/cumulative 의미를 변경하지 않는다.

### 1.3 v1에서 제외하는 기능

응답 단위 reference sampling, reference top-k 근사, learned/EMA reference, clean/V-off pairwise loss, chosen top-k 집중 항, 기존 softplus `topk_all` rejected 항의 중복 합산, GT-only V-off 억제는 제외한다. KL을 entropy 외에 임의로 추가하지 않는다. 기존 코드 경로는 그대로 남겨 비교 실험에 사용한다.

## 2. 입력, reference와 score — 기존 2절 확장

### 2.1 기호와 데이터 단위

기존과 같이 step을 \(i\), eligible GT 위치 집합을 \(T_i\), GT token을 \(y_t\), vocabulary를 \(\mathcal V\)로 둔다. GT 바로 앞 predictor 위치에서 logits를 가져온다.

\[
p^c_{\theta,t}(v)=\operatorname{softmax}(z^c_{\theta,t})_v,
\qquad
\mu_t(v)=\operatorname{softmax}(z^{\mathrm{ref},c}_t)_v.
\]

두 softmax는 **모두 전체 vocabulary**에서 계산한다. V-off policy는 \(p^o_{\theta,t}\)로 별도 표기한다.

\[
\boxed{
f_{\theta,t}(v)=\log p^c_{\theta,t}(v)-\operatorname{stopgrad}(\log\mu_t(v)).
}
\]

GT는 chosen 분포의 관측 표본이다. `orig`의 chosen은 계속 GT이며, top-k는 V-off 불확실성 항에만 적용한다. 새로운 multi-positive supervision을 암묵적으로 도입하지 않는다.

### 2.2 Frozen reference 계약

신규 `reference_mode=frozen_clean`을 정의한다. fresh run은 `reference_checkpoint`를 명시해야 한다. checkpoint는 frozen reference의 전체 유효 정책을 복원할 수 있어야 한다. adapter checkpoint라면 base model 식별자와 adapter 구성을 함께 해석한다.

reference는 다음 조건을 모두 만족해야 한다.

- 전체 학습 기간에 weight가 고정된다. `eval()` 상태와 `requires_grad=False`를 유지하고 optimizer에서 제외한다. dropout 등 학습 모드의 확률적 동작을 사용하지 않는다.
- forward는 `no_grad`로 실행하고 출력 log 확률과 expectation 가중치 모두 detach한다. reference branch를 통한 policy 입력/latent tensor로의 역전파도 금지한다.
- student와 vocabulary 크기뿐 아니라 token ID mapping, special token, tokenizer, clean 전처리 및 필요한 latent 인터페이스가 호환되어야 한다. 불일치는 시작 전에 오류로 처리한다.
- 현재 student를 매 step detach하여 reference로 사용하는 fallback, resume 시 최신 student로 reference를 다시 초기화하는 동작, EMA update를 금지한다.
- reference의 trainable storage가 student 업데이트로 바뀌어서는 안 된다. 단순한 tensor detach가 frozen model 복제를 대신하지 못한다.

reference weights의 초기 출처는 지정 checkpoint로 고정하지만, 서로 다른 입력 문맥에서 나오는 reference 확률까지 항상 같다는 뜻은 아니다.

### 2.3 같은 clean history에서 비교할 것

Reference와 policy는 같은 raw clean 입력, GT prefix, predictor 위치, attention/position 구조, latent 위치를 사용한다. Reference가 독자적으로 다른 응답이나 다른 latent history를 생성해서 비교하면 안 된다.

연속 latent가 student에서 온라인 생성되는 경로는 특히 다음 규칙을 적용한다.

1. 현재 clean 호출에 사용된 latent 입력 및 배치 정렬 정보를 replay 가능한 형태로 기록한다.
2. Reference는 해당 latent 입력을 detach하여 받아 같은 외부 conditioning history를 평가한다. 일반 token/image 처리는 frozen reference의 호환된 모듈을 사용한다.
3. Student의 clean/V-off 계산 그래프와 기존 latent gradient는 유지한다. reference 계산만 stop-gradient다.
4. Reference는 자신의 KV cache를 계산한다. 서로 다른 weight의 student KV를 reference에 재사용하지 않는다.

첨부 명세에는 이 reference replay API가 정의되어 있지 않다. 따라서 **동일 history replay 인터페이스 추가는 신규 구현의 필수 작업**이다. 이를 구현할 수 없는 backend에서는 새 alias를 거부해야 하며, 다른 history로 조용히 fallback하면 안 된다.

이 경우 이론적 의미는 각 업데이트에서 conditioning history를 고정한 token-level objective다. student-dependent latent history의 변화에 대해 reference 쪽 미분까지 포함한 전역 목적함수의 정확한 gradient라고 주장하지 않는다. Reference logit의 영구 사전 캐시는 history가 불변임을 확인할 수 없는 한 허용하지 않는다.

## 3. 이론형 DIL-BCE — 기존 3절에 신규 alias 분기 추가

### 3.1 Token-level loss

\[
\boxed{
\ell_t^+
=\operatorname{sp}\bigl(-f_{\theta,t}(y_t)\bigr)
}
\]

\[
\boxed{
\ell_t^{\mathrm{ref}}
=\sum_{v\in\mathcal V}
\operatorname{stopgrad}(\mu_t(v))
\operatorname{sp}\bigl(f_{\theta,t}(v)\bigr)
}
\]

\[
\boxed{
\ell_t^{\mathrm{orig}}=\ell_t^++\ell_t^{\mathrm{ref}}.
}
\]

Reference term에는 GT token도 포함한다. Full-vocabulary 확률 \(\mu_t\)로 가중한 합이며, vocabulary 크기로 다시 나누지 않는다. Chosen/reference 사이에 `1/2`를 넣지 않는다. GT score에 reference log 확률을 빼는 것만 구현하고 reference expectation을 생략해서도 안 된다.

다음 대체는 `dil-bce-orig`가 아니다.

| 금지된 대체 | 이유 |
|---|---|
| \(\operatorname{sp}(f_t(y_t))\) 하나를 reference 항으로 사용 | GT 표본 분포와 reference 분포가 다름 |
| V-off score를 reference 항으로 사용 | 문맥과 기대값의 분포가 모두 달라짐 |
| reference top-k 안에서 재정규화 | 원래 \(\mu_t\) expectation이 아님 |
| \(p^c_{\theta,t}\)로 reference 항을 가중 | 가중 분포가 고정 reference가 아님 |
| GT를 reference sum에서 제외 | 기준 분포와 class prior를 변경 |
| \(\operatorname{sp}(\operatorname{mean}_t f_t)\) | 정의한 token BCE의 평균과 다름 |

### 3.2 Score scale와 margin

신규 v1은 `beta=1.0`, `margin=0.0`, `margin_mode=logit`만 허용한다. 이는 현재 구현의 신규 CLI 기본 수치와도 일치하지만, 여기서는 **log density-ratio 자체를 BCE logit으로 유지하기 위한 계약**이다.

원본 alias에 arbitrary beta, 양의 margin, 독립적인 reference-term 계수를 넣지 않는다. 전체 DIL 강도는 `lambda_preference`로 조절한다. 향후 다른 score scale을 연구할 경우 별도 formula version 및 이론적 해석을 요구한다. 기존 RF alias의 beta/margin 지원 범위는 변경하지 않는다.

### 3.3 Reduction과 빈 step

선택된 위치를 \(\widetilde T_i\subseteq T_i\)라고 하면

\[
L_i^+=\frac{1}{|\widetilde T_i|}\sum_{t\in\widetilde T_i}\ell_t^+,
\qquad
L_i^{\mathrm{ref}}=\frac{1}{|\widetilde T_i|}\sum_{t\in\widetilde T_i}\ell_t^{\mathrm{ref}}.
\]

빈 \(\widetilde T_i\)이면 두 값은 student 그래프에 연결된 0으로 두고, 해당 step은 바깥 step 평균 분모에 남긴다. 전체 step 목록 자체가 비어 있으면 preference scalar는 그래프 연결 0이다. Clean CE 범위는 바꾸지 않는다.

### 3.4 Gradient 계약

Reference \(\mu_t\)가 고정된 조건에서

\[
\frac{\partial\ell_t}{\partial f_t(v)}
=\mu_t(v)\sigma(f_t(v))
-\mathbf 1[v=y_t]\sigma(-f_t(y_t)).
\]

이는 score에 대한 식이다. 실제 logit/parameter gradient에는 full softmax 및 shared parameter의 chain rule이 추가된다.

\(p^c_t=\mu_t\)이면 chosen/reference 값은 각각 \(\log2\), 합은 \(2\log2\)다. 그러나 GT 데이터 분포가 reference와 같지 않다면 gradient는 0이 아니다. 한 GT 표본에 대해 clean logits의 gradient는 이 지점에서 \((\mu_t-\mathrm{onehot}(y_t))/2\)다.

## 4. V-off top-k entropy — 기존 4절 후보 선택 재사용

### 4.1 후보 집합

기존 `preference_topk_mode`를 재사용한다.

| 모드 | \(S_t\) 선택 |
|---|---|
| `independent` | V-off logits의 top-k |
| `sync_topk` | clean logits의 top-k를 V-off에 공유 |

기본은 기존 신규 CLI와 같은 `independent`다. Vocabulary \(k\)는 사용자가 명시한다. `2 <= k < vocabulary_size`를 유지한다. 후보 ID 선택은 detach하고 선택된 V-off logit 값은 live로 둔다.

신규 옵션 `topk_include_gt`는 기본 `false`다. `true`이면 \(A_t=S_t\cup\{y_t\}\), `false`이면 \(A_t=S_t\)다. 중복 없이 구성하고 \(n_t=|A_t|\)를 위치별로 사용한다. 이 옵션은 entropy의 support만 바꾸며 DIL chosen/reference 수식에는 영향을 주지 않는다.

기존 `rejected_mode=gt`를 켜서 이 union을 구현하지 않는다. 기존 GT-union 모드는 GT 하나의 rejected score를 계산하므로 다른 loss다. 기존 top-k 후보 선택의 동점 처리 및 재현성 계약은 재사용하고 legacy 결과를 바꾸지 않는다.

### 4.2 Conditional entropy loss

\[
\log q^o_{\theta,t}(v)
=z^o_{\theta,t,v}-\log\sum_{u\in A_t}\exp z^o_{\theta,t,u},
\quad v\in A_t.
\]

\[
H_t^o=-\sum_{v\in A_t}q^o_{\theta,t}(v)\log q^o_{\theta,t}(v).
\]

\[
\boxed{
\ell_t^U=\log n_t-H_t^o
=\mathrm{KL}\left(q^o_{\theta,t}\,\middle\|\,\operatorname{Uniform}(A_t)\right).
}
\]

자연로그(natural log, nats)를 사용한다. \(\log n_t\)로 나누는 normalized entropy는 진단용으로만 기록하고 학습 loss에는 사용하지 않는다. KL의 방향은 `q_off || uniform`이며 반대 방향으로 바꾸지 않는다.

Entropy 값에 추가 softplus나 margin을 적용하지 않는다. 기존 \(\sum q\operatorname{sp}(\log q)\) 항을 같이 더하지 않는다. Full-vocabulary 확률을 그대로 top-k에 합산하는 것도 금지한다. 이 loss는 집합 내부에서 재정규화한 \(q^o\)의 entropy다.

### 4.3 Gradient와 최적해

가중치 \(q^o\)와 \(\log q^o\)는 모두 live다. 특히 `q.detach() * logq`로 entropy를 계산하면 동일한 q에서 그 token의 softmax logit gradient는 이론적으로 0이 되므로 금지한다. Reference expectation의 detached 가중치와 혼동하지 않는다. 선택된 집합이 고정되어 있으면

\[
\frac{\partial\ell_t^U}{\partial z^o_{t,v}}
=q^o_{t}(v)\left(\log q^o_t(v)+H_t^o\right),\quad v\in A_t.
\]

선택되지 않은 V-off logit에 대한 이 항 단독의 직접 미분은 0이다. Shared parameter를 통해 다른 출력이 바뀔 수 있다는 사실과 구분한다. `sync_topk`의 clean 후보 ID를 통해 clean logits로 gradient를 전파하지 않는다.

이 항의 조건부 분포 최적해는 uniform이며 값은 0이다. 거의 one-hot인 경우 값은 \(\log n_t\)에 접근한다. 이것은 최적해에 대한 성질이지 실제 softmax parameter optimization의 수렴 보장이 아니다. 극도로 포화된 softmax에서는 entropy gradient가 작아질 수 있다.

이 항은 top-k 전체의 full-vocabulary 확률 mass를 직접 제어하지 않는다.

## 5. 신규 helper와 기존 모드의 분리 — 기존 5절 확장

### 5.1 파일·함수 계약

`src/step_preference/dil_bce_orig_loss.py`를 신규 추가한다. 아래 이름은 신규 API 제안이다.

| API | 입력 | 출력/역할 |
|---|---|---|
| `dil_bce_orig_components` | clean logits `[P,V]`, reference clean logits `[P,V]`, GT IDs `[P]` | token chosen/ref losses `[P]`, GT logratio 및 필요한 진단값 |
| `topk_entropy_components` | V-off logits `[P,V]`, candidate IDs/valid mask | token entropy KL `[P]`, entropy, support 크기 |
| 기존 또는 신규 trainer reducer | token components, step ID/mask | token→step→global step 집계 |

Entropy helper는 신규 `src/step_preference/topk_entropy_loss.py`에 둔다. 기존 `dil_topk_loss.py`에서 후보 선택 코드를 공용화할 수 있지만, 공용화 전후 legacy loss와 gradient의 동등성 테스트가 전제다. 기존 softplus helper의 의미를 entropy로 바꾸지 않는다.

Reference helper는 reference logits를 내부에서도 detach하여 방어한다. 이것이 reference model freeze 검증을 대신하지는 않는다. 원 모델이 trainable 상태이면 시작 검증에서 실패해야 한다.

### 5.2 수치 안정성과 메모리

Helper는 clean/reference의 `[P,V]` shape 일치, GT ID의 정수 dtype·범위, mask와 step ID의 정렬·device를 검증한다. `ignore_index` 위치는 호출 전에 기존 target mask로 제외하고 GT gather에 음수 label이 들어가게 하지 않는다. Entropy helper의 후보 ID는 유효 범위에 있고 위치별 중복이 없어야 한다. `P=0`을 정상 입력으로 지원한다. Scalar로 반환하는 빈 loss는 student 그래프에 연결하고 reference 그래프에는 연결하지 않는다.

Loss 계산은 FP32 `log_softmax`, `logsumexp`, 안정적인 softplus를 사용한다. `log(softmax(...))`와 확률의 직접 나눗셈 \(p/\mu\)를 피한다. logits에는 NaN/Inf가 없어야 한다. BF16/FP16 model output은 loss 계산 시 FP32로 승격한다.

Reference expectation은 full vocabulary를 유지한다. FP32 exp의 underflow 가능성을 이유로 arbitrary probability floor를 넣어 target을 바꾸지 않는다. 유한한 logits에서 log 확률은 log-softmax로 유지하고 probability weight만 exp한다.

GT union의 ragged padding은 valid mask로 처리한다. Padding의 `0 * (-inf)`가 entropy에 NaN을 만들지 않도록 masked log 값은 곱셈 전에 안전하게 처리한다. 유효 후보 수를 padding 길이로 잘못 계산하지 않는다.

기존 top-k helper의 위치 단위 chunking 방식을 참고하여 신규 기본 `loss_chunk_positions=128`을 둔다. Chunk별 평균을 다시 평균하지 말고 token 합과 step별 count를 누적한다. Clean/reference의 full-vocabulary log-normalizer와 gradient를 유지해야 한다. Vocabulary를 chunking하더라도 truncated softmax로 변경하거나 clean log-normalizer를 detach하면 안 된다.

미세한 부동소수점 오차로 entropy KL이 작은 음수가 될 수 있다. 학습 loss를 무조건 clamp하여 목적을 변경하지 말고, 허용 오차 내인지 검사한다. 진단 표시용 clamp는 raw 값과 구분한다.

## 6. Total loss, mass KL와 distributed reduction — 기존 6절 수정 방향

### 6.1 Total loss

\[
L_i^U=\operatorname{mean}_{t\in\widetilde T_i}\ell_t^U.
\]

\[
\boxed{
L_{\mathrm{total}}
=\lambda_{\mathrm{CE}}L_{\mathrm{clean\ CE}}
+\lambda_P\operatorname{mean}_i(L_i^++L_i^{\mathrm{ref}})
+\lambda_U\operatorname{mean}_iL_i^U
+\lambda_M\operatorname{mean}_iL_i^{\mathrm{mass}}.
}
\]

Mapping은 `lambda_ce` → \(\lambda_{CE}\), `lambda_preference` → \(\lambda_P\), **신규 `lambda_topk_entropy`** → \(\lambda_U\), 기존 mass 계수 → \(\lambda_M\)다. Entropy는 `lambda_preference` 안에 다시 곱하지 않는다.

기존 `lambda_entropy`는 entropy-margin 계열과의 혼동을 막기 위해 새 alias에서도 0을 요구한다. Entropy token selection 옵션과도 별개다.

### 6.2 Optional mass KL

기본 mass 계수는 0이다. Entropy 활성화와 함께 기존 clean→V-off Bernoulli mass KL을 명시적으로 활성화할 수 있다. 동일한 \(A_t\)에 대해

\[
M_t^c=\sum_{v\in A_t}p_t^c(v),\quad
M_t^o=\sum_{v\in A_t}p_t^o(v),
\]

\[
L_i^{\mathrm{mass}}
=\operatorname{mean}_{t\in\widetilde T_i}
\mathrm{KL}\bigl(\mathrm{Bern}(\operatorname{stopgrad}(M_t^c))\|\mathrm{Bern}(M_t^o)\bigr).
\]

기존 수식과 stop-gradient 방향을 유지한다. Reference model은 이 항에 참여하지 않는다. v1에서는 `lambda_topk_chosen_mass=0`만 허용한다. Mass 계수가 양수이면 entropy도 활성화되어 있어야 한다. \(A_t=\mathcal V\)이면 mass KL과 해당 미분은 정확히 0으로 처리한다.

### 6.3 Step 평균과 DDP

신규 alias도 긴 step에 추가 가중치를 주지 않는다. Selection으로 비어 있는 step을 포함한 original step 목록을 바깥 분모로 사용한다.

새 reducer는 gradient가 global step 평균과 일치하도록 해야 한다. Rank 수 \(W\), rank \(r\)의 step 수 \(N_r\), 전체 수 \(N=\sum_rN_r\)에 대해, DDP가 gradient를 rank 평균한다면 각 rank의 preference contribution을

\[
\frac{W}{N}\sum_{i\in r}L_i
\]

로 구성하면 global step 평균과 일치한다. Rank별 step 수가 같으면 기존 local step mean과 동일하다. 이 처리는 새 alias 전용으로 추가하거나, 기존 reducer의 같은 계약이 확인되면 재사용한다. Clean CE scalar의 기존 reduction을 임의로 이 방식에 함께 바꾸지 않는다.

빈 rank 또는 전역 빈 target에서도 필요한 student 계산 그래프와 scalar의 device/dtype을 보존한다. Reference는 언제나 gradient-free다.

## 7. Entropy 기반 token selection — 기존 7절 재사용 범위

신규 alias의 entropy loss와 entropy token selection은 별개다.

기본은 `preference_entropy_top_percent=None`으로 기존 eligible token 전체를 사용한다. 활성화할 때는 기존 온라인 source `clean`, `voff`, `gap`, `intersection`의 의미, sample 단위 순위, ceil 비율, 동점 처리, 교집합 크기, 빈 step 규칙을 유지한다.

Chosen, reference, entropy, optional mass에 **동일한 선택 mask**를 적용한다. Clean CE에는 이 mask를 적용하지 않는다. 따라서 selection 활성화 시 DIL이 학습하는 문맥의 분포도 선택된 위치로 제한된다는 점을 명시한다.

기존 `image_gap` artifact 경로는 reference-free GT-union 등 특정 조합에 제한되어 있으므로 신규 v1에서는 지원하지 않고 명시적으로 오류를 낸다. 검증 없이 제한을 해제하지 않는다.

Entropy 계수가 0이어도 selection source가 V-off를 요구하면 V-off forward가 필요하다. `clean` source 또는 selection 비활성이고 entropy/mass도 비활성이면 V-off loss 경로는 실행하지 않는다.

## 8. Forward, trainer와 reference lifecycle — 기존 8절 확장

### 8.1 필요한 forward 결정

`need_reference=true`는 새 alias에서 항상 성립한다. `need_off`는 다음 필요 중 하나라도 있을 때만 true다: top-k entropy 활성화, 허용된 mass 활성화, V-off 기반 token selection.

추가 entropy-only 모델 forward는 만들지 않는다. 기존 policy clean/V-off logits를 재사용한다. Frozen reference의 **clean evaluation**만 새로 추가한다. 기존 `shared_prefix_3pass` 이름을 정확한 호출 횟수로 해석하지 않는다.

### 8.2 Trainer 처리 순서

1. Alias를 canonicalize하고 신규 alias 전용 기본값을 resolve한 후 validate한다.
2. 기존 target/span/predictor plan을 생성하고 `need_off`를 결정한다.
3. Policy clean forward와 필요한 V-off replay를 기존 개입 의미로 수행한다. Clean CE scalar를 보존한다.
4. 기존 규칙으로 selection mask를 만들고 clean history/reference alignment 정보를 구성한다.
5. 같은 clean history를 frozen reference에 no-grad replay하여 필요한 predictor의 full-vocabulary logits를 얻는다.
6. 신규 helper로 clean token chosen/reference BCE를 계산한다. 기존 `_mean_logp` 후 step softplus helper에 넘기지 않는다.
7. Entropy가 활성화되면 후보 집합을 만들고 V-off entropy를 계산한다. Optional mass는 같은 후보 집합으로 계산한다.
8. Token→step→global step reduction과 total 가중합을 수행한다.
9. Raw/weighted component와 진단값을 분리하여 기록하고 finite 검사를 수행한다.

모든 rank에서 DDP/FSDP/collective 호출 순서가 일치해야 한다. 선택된 위치가 없는 rank를 이유로 필요한 distributed forward/collective를 rank별로 임의 생략하지 않는다.

### 8.3 Reference 저장과 재개

Fresh run에서 지정 checkpoint의 reference를 로드하여 run의 immutable `reference_snapshot` 또는 그와 동등한 content-addressed artifact로 보존한다. Student checkpoint마다 큰 reference weights를 복제할 필요는 없지만, reference artifact와 manifest가 함께 복원 가능해야 한다.

Manifest에는 실제 weight/config의 digest, base/adapter 조합, tokenizer와 token mapping, architecture/revision, 필요한 전처리 및 replay 호환성 정보를 포함한다. 경로 문자열만 같다고 동일 reference로 판정하지 않는다. 저장소의 기존 artifact hash 체계가 있으면 재사용한다.

Resume은 저장된 reference artifact를 복원한다. 사용자 override가 있으면 content fingerprint가 일치해야 한다. Artifact 누락/불일치는 오류다. 최신 student나 base model로 조용히 재초기화하지 않는다. `trainer.train()` 호출 뒤에도 reference가 eval 상태인지 확인한다.

## 9. 설정, 기본값과 validation — 기존 9절에 selector별 분기

### 9.1 신규 alias의 resolved 설정

표의 신규 이름은 이번 구현의 인터페이스 계약이다. 기존 문서가 vocabulary k 및 mass의 전체 CLI 철자를 제공하지 않았으므로, 아래 `preference_topk_k`와 `lambda_topk_mass`는 **명시적인 신규 공개 alias 이름**으로 제안한다. 저장소의 기존 해당 필드에 1:1 매핑하고 동일 의미의 상태를 중복 저장하지 않는다. 기존 철자와 함께 제공되면 충돌 검증을 한다.

| 설정 | 새 alias의 기본/제약 | 성격 |
|---|---|---|
| `loss_type` | `dil-bce-orig` → `dil_bce_orig` | 신규 alias |
| `reference_mode` | `frozen_clean` | 신규 enum 분기 |
| `reference_checkpoint` | fresh run에서 필수, resume은 manifest 복원 | 신규 |
| `reference_context_mode` | `shared_clean_history`만 | 신규 |
| `reference_expectation` | `full_vocab`만 | 신규 |
| `beta` / `margin` / `margin_mode` | `1.0` / `0.0` / `logit` 고정 | 기존 필드, 신규 제약 |
| `chosen_mode` | `gt`만 | 기존 필드 |
| `chosen_reduction` | `token_mean`만 | 기존 필드, 신규 기본값 |
| `rejected_mode` | resolved `null`; 명시적 `gt`/`topk_all`는 오류 | 원래 rejected 경로 비활성 |
| `lambda_ce` | `0.1`, finite non-negative | 기존 신규 CLI 기본값 유지 |
| `lambda_preference` | `1.0`, finite positive | DIL 항 필수 |
| `lambda_topk_entropy` | `0.0`, finite non-negative | 신규, \(\lambda_U\) |
| `lambda_entropy` | `0.0`만 | legacy entropy-margin 비활성 |
| `preference_topk_k` | 기본 없음; entropy 양수이면 명시 필수 | 기존 vocabulary k에 매핑 |
| `preference_topk_mode` | `independent`; `sync_topk` 허용 | 기존 의미 유지 |
| `topk_include_gt` | `false`; `true` 허용 | 신규 entropy support 옵션 |
| `lambda_topk_mass` | `0.0`; 양수이면 entropy도 양수 | 기존 mass 계수에 매핑 |
| `lambda_topk_chosen_mass` | `0.0`만 | chosen mass 역방향 비활성 |
| `preference_entropy_top_percent` | `None` | 기존 selection |
| `loss_chunk_positions` | `128`, 양의 정수 | 신규 helper 자원 설정 |

수치 기본값은 명세상의 시작값이지 성능 최적값이 아니다. `lambda_ce=0`도 허용하며 이 경우 entropy가 꺼져 있으면 total loss 자체가 이론형 token DIL-BCE다.

### 9.2 Validation 규칙

Alias별 default resolution을 기존 공통 dataclass default와 분리한다. 새 alias에 legacy default `reference=none`, `rejected_mode=topk_all`, `chosen_reduction=mean_logp`가 주입되지 않게 한다. 사용자가 명시적으로 준 충돌 옵션은 무시하지 말고 오류를 낸다.

`dil_bce_orig`의 fresh run은 유효한 frozen reference가 필수다. `reference_mode=none`, nonzero margin, beta≠1, pairwise score transform, sequence/mean-logp reduction, reference top-k truncation, chosen top-k는 거부한다.

Entropy 양수이면 V-off를 지원하는 기존 visual-only clean-prefix 조합, 유효한 vocabulary k, 같은 tokenizer/positions가 필요하다. 기존에 금지한 `remove_full_visual_block` 등 조합을 이 작업으로 허용하지 않는다. Image replacement의 기존 branch 제약을 유지한다.

Entropy 비활성이면 후보 k가 설정되어 있더라도 관련 tensor 연산은 수행하지 않는다. 비활성 옵션이라는 resolved 정보를 로그에 남긴다. 원본 DIL과 RF checkpoint는 서로 loss alias를 바꿔 resume하지 않는다.

### 9.3 설정 예시 — 구현 후 목표 인터페이스

다음은 새 resolver가 받아야 할 설정 예시다. 현재 실행 가능한 CLI 명령이나 기존 YAML loader 지원을 확인했다는 뜻은 아니다. 실제 entrypoint에서 이 필드들을 연결한다.

**기본 DIL-BCE + 기존 clean CE:**

```yaml
loss_type: dil-bce-orig
reference_mode: frozen_clean
reference_checkpoint: /path/to/frozen-reference-checkpoint
reference_context_mode: shared_clean_history
reference_expectation: full_vocab
chosen_mode: gt
chosen_reduction: token_mean
beta: 1.0
margin: 0.0
margin_mode: logit
lambda_ce: 0.1
lambda_preference: 1.0
lambda_topk_entropy: 0.0
lambda_entropy: 0.0
```

**위 설정에 V-off top-k entropy 추가:**

```yaml
lambda_topk_entropy: 0.1
preference_topk_k: 32
preference_topk_mode: independent
topk_include_gt: false
lambda_topk_mass: 0.0
lambda_topk_chosen_mass: 0.0
```

`0.1`과 `32`는 설정 연결을 확인하기 위한 예시이며 최적 계수/k 추천이 아니다. Entropy GT union 비교는 `topk_include_gt: true`만 변경한다. 기존 `rejected_mode=gt`를 추가하면 validation 오류다.

## 10. 기록, formula identity, checkpoint와 migration — 기존 10절 확장

### 10.1 로그

| Key | 의미 |
|---|---|
| `loss/dil_bce_orig_chosen` | raw chosen BCE의 step 평균 |
| `loss/dil_bce_orig_reference` | raw reference expectation의 step 평균 |
| `loss/dil_bce_orig_sum` | 위 두 값의 합 |
| `loss/dil_bce_orig_weighted` | `lambda_preference` 적용값 |
| `loss/topk_entropy_kl` | raw \(\log n-H(q^o)\)의 step 평균 |
| `loss/topk_entropy_weighted` | `lambda_topk_entropy` 적용값 |
| 기존 `loss/ce`, `loss/total` | 기존 의미 유지 |
| 기존 mass 로그 | 신규 alias에서도 동일한 수식·방향으로 기록 |
| `stats/orig_policy_gt_logp` | clean full-vocabulary GT log 확률 |
| `stats/orig_reference_gt_logp` | reference clean GT log 확률 |
| `stats/orig_gt_logratio` | 위 두 값의 차이 |
| `stats/topk_entropy_off` | \(H(q^o)\) |
| `stats/topk_entropy_off_normalized` | \(H(q^o)/\log n\), 진단 전용 |
| `stats/topk_support_size` | 실제 \(n_t\) |
| `stats/topk_gt_coverage_pre_union` | union 전 GT 포함률 |
| `stats/topk_mass_clean`, `stats/topk_mass_off` | 조건부 entropy와 구분한 full-vocabulary mass |
| `stats/preference_selected_tokens/steps` | selection과 빈 step 해석용 count |
| `stats/reference_prob_sum_error` | full-vocabulary reference weight 합과 1의 차이 |

Reference term을 `loss/dil_bce_negative_gt` 또는 `negative_topk`로 기록하지 않는다. 해당 legacy key를 다른 의미로 재사용하지 않는다. Token 진단 평균과 학습의 step 평균을 구분하여 metadata에 남긴다.

### 10.2 Formula identity

다음 신규 formula version을 제안한다.

| 모드 | `preference_formula_version` |
|---|---|
| entropy 꺼짐 | `dil_bce_orig_token_ref_full_vocab_v1` |
| entropy 켜짐, GT union 없음 | `dil_bce_orig_token_ref_full_vocab_topk_entropy_v1` |
| entropy 켜짐, GT union 있음 | `dil_bce_orig_token_ref_full_vocab_topk_gt_union_entropy_v1` |

추가 metadata는 canonical alias, reference fingerprint/context/expectation, token reduction, beta/margin, 모든 loss 계수, entropy support/mode/k, optional mass 방향, selection, target span, ablation/branch, tokenizer/전처리를 포함한다. `formula_domain=teacher_forced_token`을 저장하여 sequence-level loss와 혼동하지 않게 한다.

### 10.3 Schema와 resume

기준 명세의 schema는 21이다. 대상 저장소도 21이면 이번 기능의 schema를 22로 올린다. 별도 변경이 이미 번호를 사용했다면 충돌 없는 다음 번호를 사용한다. 현재 저장소 상태를 확인하지 않고 22가 비어 있다고 가정하지 않는다.

기존 checkpoint가 새 entropy/reference 필드를 갖고 있지 않으면 **기존 alias에는** 기존 의미대로 `reference=none`, 신규 entropy=0을 보완한다. 이것은 RF→orig loss migration이 아니다.

새 alias의 resume에서는 reference fingerprint 및 formula-affecting metadata가 같아야 한다. Alias의 hyphen/underscore 표기 차이만 허용한다. 사용자가 명시적으로 다른 entropy 계수/k/union/reduction을 주면 오류를 낸다. 다른 objective 비교는 weight-only warm start로 새 run을 시작한다. 이전 optimizer를 다른 loss로 자동 연결하지 않는다.

## 11. 구현 순서, 테스트와 검증 한계 — 기존 11절 대체가 아닌 추가

### 11.1 수정 파일 지도

| 파일/영역 | 변경 범위 |
|---|---|
| `src/step_preference/config.py` | alias, selector별 defaults, reference/entropy validation, formula identity 입력 |
| `src/step_preference/dil_bce_orig_loss.py` — 신규 | full-vocabulary chosen/reference token BCE |
| `src/step_preference/topk_entropy_loss.py` — 신규 | conditional entropy KL 및 diagnostics |
| `src/stage2/trainer.py` | orig 분기, reference replay 호출, 선택 mask, 신규 reduction/로그 |
| 실제 model/trainer 초기화 경로 — 위치 확인 필요 | frozen reference 로딩·수명주기·optimizer 제외 |
| `src/step_preference/packing.py` 및 기존 model replay 경로 | 필요한 경우 동일 clean-history reference replay 계약 제공 |
| `src/stage2/checkpointing.py` | reference manifest, metadata, resume 검증 |
| 기존 CLI entrypoint — 위치 확인 필요 | 공개 alias/필드 연결, legacy defaults 보존 |
| 신규 unit/integration tests | 아래 acceptance criteria |

첨부 명세에서 위치를 알 수 없는 loader/entrypoint는 임의 경로를 만들어 지정하지 않는다. 작업 시작 시 실제 함수 위치를 확인하고 이 표의 해당 행만 구체화한다. 기존 reference 설계 문서가 저장소에 있으면 이번 계약과 차이를 대조하되, reference expectation을 생략하는 설계로 후퇴하지 않는다.

권장 작업 순서는 loss oracle → 신규 helper → alias/default validation → frozen reference/history replay → trainer 연결 → checkpoint/logging → legacy regression 및 integration 검증이다. 실제 구현에서는 각 기능의 failing test를 먼저 정의하고 구현 후 통과 여부를 기록한다.

### 11.2 필수 수치·gradient 테스트

| ID | 검증 | 합격 조건 |
|---|---|---|
| D1 | 작은 vocabulary FP64 oracle | chosen/ref/total이 직접 수식 계산과 일치 |
| D2 | reference expectation | full vocabulary, GT 포함, 추가 \(1/V\) 없음 |
| D3 | \(p=\mu\) | chosen=ref=log2, total=2log2; 값과 gradient를 혼동하지 않음 |
| D4 | \(p=\mu\), 하나의 GT | logit gradient=(mu−onehot)/2 |
| D5 | 기대 chosen 분포 \(p_+=a\) | 양의 toy \(a\)에서 \(p=a\)의 population gradient가 0 |
| D6 | reference detach/freeze | reference weight·logit·reference 입력경로 gradient 없음; student gradient 있음 |
| D7 | 잘못된 근사 방지 | GT-only, top-k reference, live-policy weighting 구현이 oracle과 다름을 검출 |
| D8 | reduction | token BCE 평균과 평균 score BCE가 다른 fixture로 신규 경로 확인 |
| E1 | uniform 후보 logits | entropy=log n, KL≈0, entropy logit gradient≈0 |
| E2 | nonuniform 후보 | KL>0, analytic gradient와 autograd/finite difference 일치 |
| E3 | near one-hot | KL이 log n에 접근하고 NaN/Inf가 없음 |
| E4 | live q | nonuniform fixture에서 올바른 gradient는 nonzero; `q.detach()*logq` variant는 약 0임을 검출 |
| E5 | 후보 집합 | independent/sync, GT union 중복 제거, n=k/k+1을 검증 |
| E6 | support gradient | entropy 단독의 off 비선택 logit 미분=0, sync clean selector 미분=0 |
| E7 | 전체 vocabulary union | n=V 처리; mass KL 및 gradient=0 |
| N1 | chunking | chunk 1/128/전체의 loss와 gradient가 허용오차 내 일치 |
| N2 | dtype/극단 logits | FP32 helper와 FP64 oracle 비교, NaN/Inf 없음 |
| N3 | ragged union padding | 0×(−inf) 오염 없음, padding을 support로 세지 않음 |
| N4 | 불법 설정 | beta/margin/reduction/ref/tokenizer 조합 오류 검출 |

Floating point 비교 기준은 CPU FP64 oracle에서 예를 들어 `atol=1e-10, rtol=1e-8`, FP32 helper 비교에서는 `atol=1e-5, rtol=1e-4`로 시작한다. 극단 fixture와 모델 integration에는 dtype에 맞는 근거 있는 tolerance를 별도로 문서화한다. 무조건 tolerance를 키워 통과시키지 않는다.

수치 fixture 예시: \(p^c=(0.5,0.3,0.2)\), \(\mu=(0.2,0.5,0.3)\), GT ID=0이면 chosen 항은 `0.336472236621213`, reference 항은 `0.638802095451739`, 합은 `0.975274332072952`다. 여기에는 CE/entropy/step 가중치가 포함되지 않는다. 이 값은 문서 수식의 독립적인 소규모 계산 예시이며 프로젝트 구현 테스트 통과를 의미하지 않는다.

### 11.3 필수 integration·호환성 테스트

| ID | 검증 | 합격 조건 |
|---|---|---|
| I1 | alias normalization | hyphen/underscore가 같은 formula·resolved config |
| I2 | reference lifecycle | optimizer step 뒤 ref weight 불변, train 호출 뒤에도 ref eval |
| I3 | history alignment | 같은 raw input/GT/predictor/latent history 사용; 잘못된 replay는 fail |
| I4 | reference identity fixture | 동일 weight·eval·입력에서 policy/ref logits 일치 |
| I5 | entropy=0 | 순수 orig preference 경로, 필요 없는 off/entropy forward 없음 |
| I6 | off 기반 selection | entropy=0이어도 필요한 off forward 유지 |
| I7 | 빈 mask/빈 step | 그래프 연결 0, 원래 step 분모 유지, CE 범위 불변 |
| I8 | DDP reduction | 동일/상이한 rank별 step 수가 single-process global step oracle과 일치 |
| I9 | resume | reference 누락/변경, formula 변경 거부; 동일 설정 재현 |
| I10 | legacy regression | RF/GT union/topk_all 수식·gradient·defaults·migration 결과 불변 |
| I11 | total 구성 | old V-off BCE/entropy-margin이 중복 합산되지 않음 |
| I12 | 기록 | reference와 rejected 로그 분리, raw/weighted scale 정확 |

실제 GPU smoke test에서는 finite loss/gradient, reference 불변, 메모리 사용량, 추가 reference 계산 비용을 기록한다. clean 정확도나 시각적 추론 성능 향상은 별도 학습·평가 결과 없이는 완료 조건이나 보장으로 쓰지 않는다.

### 11.4 완료 기준

이 명세의 수식, freeze/history 계약, 신규 alias/defaults, reference artifact 재개, entropy gradient, legacy regression이 구현·검증되어야 기능 완료로 표시한다. 현재 문서는 설계 산출물이다. 실제 저장소 패치, 모델 forward, 테스트 통과 또는 학습 성능을 주장하지 않는다.

## 부록 A. 이론적 설명의 범위

원 DIL의 chosen/reference BCE 구조는 [T1]의 density-ratio estimation을 따른다. 이번 신규 구현은 이를 고정된 teacher-forced conditioning history의 token 분포에 적용한다. Reference expectation은 해당 token의 전체 vocabulary에 대해 계산한다.

DIL-BCE와 V-off entropy의 결합은 “clean imitation + intervention-context uncertainty regularization”이다. 이론적 reward를 \(\log(p_+/\mu)\)로 두는 KL-regularized RL 해석과 연결할 수 있지만, BCE와 reverse-KL은 같은 목적함수가 아니다. Shared parameter와 추가 entropy/CE/mass 조건 아래에서 두 목적의 최적해가 항상 같다는 주장은 하지 않는다.

V-off uniform은 추가한 설계 가정이다. 언어 문맥만으로 해결 가능한 token까지 무조건 불확실하게 만드는 것이 언제나 유익하다고 가정하지 않는다. 본 작업은 그 가정을 구현·분리해 검증 가능하게 하는 것이며, 단일 GT를 임의 top-k positive로 대체하는 작업이 아니다.

## 부록 B. 기존 명세에 반영할 편집 체크리스트

| 기준 문서 절 | 편집 |
|---|---|
| 1. 핵심/범위 | selector 표에 `dil-bce-orig` 추가; reference-free 제한을 legacy selector로 한정 |
| 2. 입력/GT 점수 | ref clean logits와 logratio 추가; 신규 token reduction의 예외 명시 |
| 3. 기본 RF 수식 | 수식 보존; 신규 orig 수식을 별도 하위절로 추가 |
| 4. GT-union | 기존 수식/제약 보존; entropy support union과 구분 |
| 5. topk_all | 기존 live-q softplus 수식 보존; 신규 entropy helper를 별도 명시 |
| 6. total/mass | selector별 total 식과 독립적인 lambda_topk_entropy 추가 |
| 7. entropy selection | 신규 alias 재사용 범위와 image_gap 미지원 명시 |
| 8. forward | frozen clean reference replay/lifecycle 추가; 기존 개입 의미 유지 |
| 9. 설정 | alias별 defaults/validation, beta=1·margin=0·token_mean 규칙 추가 |
| 10. checkpoint | orig formula/reference fingerprint/migration 분기 추가 |
| 11. 확인 범위 | 실제 구현·테스트가 끝난 항목만 검증 결과로 갱신 |

## 근거와 기준 파일

[S1] `dil-bce_current.md`, 「DIL-BCE v1 — 현재 구현 명세」, 기준일 2026-10-07. 본 문서의 기존 기능·파일 경로·설정·reduction·개입·checkpoint 설명은 이 첨부 명세를 근거로 한다. 실제 저장소를 직접 확인한 결과가 아니다.

- 기준 파일 크기: 21,441 bytes.
- SHA-256: `68569352d2677cb918b1159d008ba9b65b3d8b8189b650494c54c5cd8c97efab`.
- 이 문서 작성 과정에서 기준 파일은 변경하지 않았다.

[T1] Teng Xiao, Yige Yuan, Mingxiao Li, Zhengyu Chen, Vasant G Honavar. *On a Connection Between Imitation Learning and RLHF*. arXiv:2503.05079v1, 2025. §4.3–4.4, Table 1, Appendix A의 BCE 식. 원 논문의 이론 수식 확인에만 사용하며, 이번 저장소의 구현 상태나 성능 근거로 사용하지 않는다.

나머지 신규 alias, full-vocabulary token 구현, entropy 계수, 기본값, reference replay 계약과 tests는 이 문서의 제안 사항이다.
