# Preference prefix 선택

기존 `clean_prefix` 구현과 기본값은 유지한다. 새 `v_off_prefix`는 선택적으로만 사용한다.

| 설정 | Preference 계산 | 원본 이미지 접근 | START/END |
|---|---|---|---|
| `clean_prefix` | 기존 clean 점수와 V-off suffix 점수 비교 | 기존과 동일 | 기존 별도 옵션 |
| `v_off_prefix`, image block=false | 같은 prefix를 다시 계산하여 현재 latent ON/OFF 비교 | 두 쪽 모두 허용 | 둘 다 유지 |
| `v_off_prefix`, image block=true | 같은 prefix를 다시 계산하여 현재 latent ON/OFF 비교 | 두 쪽 모두 차단 | 둘 다 유지 |

새 방식에서는 clean KV를 재사용하지 않는다. 이미지 차단은 모든 preference query에서 원본 image token key를 가리므로, 앞선 텍스트의 KV를 통한 **원본 image token의 재유입**도 막는다. 다만 clean rollout이 생성한 latent 벡터와 이전 텍스트 자체는 그대로다. 따라서 모든 시각 정보를 지우는 실험이 아니라, **직접 이미지 접근 없이 현재 latent가 추가로 도움이 되는지** 보는 실험이다. 이미지 경계 토큰, 위치 번호, 텍스트 토큰은 삭제하지 않는다.

OFF에서는 현재 latent key만 가린다. START/END는 가리지 않고, END hidden state도 각 조건에서 다시 계산한다. ON/OFF는 같은 dropout 난수를 쓴다. 이 난수 공유는 새 방식에만 적용한다. 기존 방식과 새 이미지-허용 방식의 loss/gradient 일치 검증은 dropout=0 조건이다.

목표 $y$는 두 쪽에서 동일하다. `immediate_text`는 latent 바로 다음 **데이터 reasoning entry 한 개**이며, 한 토큰이나 마침표 기준 문장이 아니다. `until_next_latent` 옵션도 유지한다. 이번 비교 실험은 `immediate_text`로 고정한다.

$$
s_{\mathrm{on/off}}=\frac{1}{|y|}\sum_{t=1}^{|y|}
\log p_\theta(y_t\mid C_{\mathrm{on/off}},y_{<t}),\qquad
\mathcal L_{\mathrm{pref}}=-\log\sigma\bigl(\beta(s_{\mathrm{on}}-s_{\mathrm{off}}-m)\bigr).
$$

CE는 원본 clean 학습 경로를 그대로 쓴다. 이번 실험은 $\mathcal L=\mathcal L_{\mathrm{CE}}+10\mathcal L_{\mathrm{pref}}$, $\beta=0.1$, $m=0$, entropy weight=0이다. 별도의 reference model을 새로 도입하지 않는다. 이 objective를 reference-model DPO로 부르지 않는다.

```bash
export STEP_PREFERENCE_TARGET_SCOPE=immediate_text
export STEP_PREFERENCE_PREFIX_MODE=v_off_prefix
export STEP_PREFERENCE_BLOCK_ORIGINAL_IMAGE=false  # 비교 조건에서는 true
export STEP_VISUAL_ONLY_CONTEXT_PREFERENCE=true
export STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES=false
```

새 prefix는 V-only context preference와 START/END 유지 조건만 지원한다. 맞지 않는 조합은 오류로 중단한다. 기존 방식으로 돌아가려면 prefix=`clean_prefix`, image block=`false`로 설정한다. Python CLI는 `--step_preference_prefix_mode`와 `--step_preference_block_original_image`다.

새 방식의 `stats/logp_evidence_on/off`가 비교 점수다. 호환용 `stats/logp_clean`은 새 방식에서 evidence-on 점수의 별칭이며, 이미지 차단 시 **clean CE 경로 점수라는 뜻이 아니다**. Raw preference loss와 ON/OFF 차이를 함께 보고, loss 감소만으로 정답률 상승이나 장기 수렴을 주장하지 않는다.
