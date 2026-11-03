# Ours1 Stage 2B Preference Objective

이 문서는 `ours1` Stage 2B의 CE+Preference variant에서 사용하는
context-preference objective를 정리한다.

## Step score

각 reasoning step `s`의 동일한 정답 token sequence `y_s`에 대해 다음 세
가지 length-normalized score를 계산한다.

$$
c_s
=
\frac{1}{|y_s|}
\sum_t
\log p_\theta(y_{s,t} \mid C_{\mathrm{clean}})
$$

$$
v_s
=
\frac{1}{|y_s|}
\sum_t
\log p_\theta(y_{s,t} \mid C_{\mathrm{V\text{-}off}})
$$

$$
t_s
=
\frac{1}{|y_s|}
\sum_t
\log p_\theta(y_{s,t} \mid C_{\mathrm{T\text{-}off}})
$$

- `c_s`: 전체 clean context에서 계산한 정답 step의 평균 log-probability
- `v_s`: 해당 step의 latent 구간을 attention에서 차단한 V-off 평균
  log-probability
- `t_s`: 해당 step의 이전 textual context 구간을 attention에서 차단한
  T-off 평균 log-probability

세 점수는 모두 동일한 정답 token을 대상으로 한다. V-off와 T-off는 별도의
rejected answer를 생성하지 않고, clean sequence의 일부 context에 대한
attention만 차단한다.

## V-off preference loss

V-off에서는 clean score가 visual context를 차단한 score보다 margin `m`만큼
높아지도록 학습한다.

$$
L_V
=
\frac{1}{S}
\sum_s
-\log \sigma\left(
\beta\left[(c_s-v_s)-m\right]
\right)
$$

## T-off preference loss

T-off에서는 clean score가 이전 textual context를 차단한 score보다 margin
`m`만큼 높아지도록 학습한다.

$$
L_T
=
\frac{1}{S}
\sum_s
-\log \sigma\left(
\beta\left[(c_s-t_s)-m\right]
\right)
$$

## Preference aggregation

V-off와 T-off loss는 동일한 비중으로 평균한다.

$$
L_{\mathrm{Preference}}
=
\frac{L_V+L_T}{2}
$$

CE+Preference variant의 전체 objective는 다음과 같다.

$$
L_{\mathrm{total}}
=
L_{\mathrm{CE}}
+
\lambda_{\mathrm{pref}}L_{\mathrm{Preference}}
$$

현재 실험 설정은 다음과 같다.

| Parameter | Value |
|---|---:|
| `beta` | `0.1` |
| Preference margin `m` | `0.0` |
| `lambda_pref` | `1.0` |
| Entropy weight | `0.0` |

따라서 현재 CE+Preference objective는 다음과 같다.

$$
L_{\mathrm{total}}
=
L_{\mathrm{CE}}
+
\frac{L_V+L_T}{2}
$$

## Gradient flow

Clean score와 V-off/T-off score는 모두 같은 학습 모델에서 계산하며 양쪽
경로 모두 gradient가 연결되어 있다. Context-preference objective에서는
reference model score를 빼거나 off score를 detach하지 않는다. 따라서 이
objective는 reference model과 생성된 rejected answer를 사용하는 legacy DPO
objective와 다르다.

## Implementation references

- Preference loss: `src/step_preference/dpo_loss.py::context_preference_loss`
- Modality aggregation and total loss: `src/stage2/trainer.py::_reduce_context_objective`
- V-off/T-off context ranges: `src/step_preference/sidecar.py::_chosen_branch`
- Clean and sidecar score construction:
  `src/stage2/trainer.py::_compute_context_preference_loss`
