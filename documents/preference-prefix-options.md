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

## 별도 모드: `v_off_insertion`

위 절의 `v_off_prefix`는 기존 실험의 의미를 유지한다. `v_off_insertion`은 실제 이미지 없는 OFF를 먼저 실행하는 별도 선택값이다. OFF에서는 모든 원본 `vision_start/image_pad/vision_end` 블록과 latent pad 슬롯을 삭제하며, 텍스트와 latent START/END는 유지한다. OFF에서 latent를 생성하지 않는다.

ON은 실제 OFF의 현재 START 직전 KV를 미분 가능하게 재사용한다. 해당 START 직전에 원본 이미지 블록 전체를 기존 순서로 삽입하고, 그 분기의 문맥에서 기존 Stage2 recurrence로 latent를 생성한다. 이미지 encoder의 기존 frozen/no_grad 설정은 바꾸지 않는다. clean decoder의 latent/hidden/KV는 새 scorer에 넘기지 않는다. 삽입 이미지에는 이후 텍스트 채점에서도 causal attention으로 접근하며 추가 이미지 차단은 없다.

| `step_preference_branch_mode` | 두 번째 target의 ON 문맥 |
|---|---|
| `independent` (기본) | `Q T0 S E T1 I S Z2 E T2` — 매번 원래 OFF에서 분기 |
| `cumulative` | `Q T0 I S Z1 E T1 I S Z2 E T2` — 앞선 삽입·latent·텍스트 이력 유지 |

OFF는 두 모드 모두 `Q T0 S E T1 S E T2`다. target이 없는 latent 블록도 OFF에서는 비우며, ON에서는 활성 분기로 지정한 블록만 복원한다. 원본 이미지가 여러 개면 각 활성 시점에 전체 블록 집합을 반복 삽입한다. 위치 번호와 predictor 위치는 실제 삭제/삽입 배열에서 다시 계산한다.

```bash
export STEP_PREFERENCE_ENABLED=true
export STEP_PREFERENCE_MODE=stage2b
export STEP_PREFERENCE_OBJECTIVE=context_preference
export STEP_SHARED_PREFIX_ENABLED=true
export STEP_PREFERENCE_PREFIX_MODE=v_off_insertion
export STEP_PREFERENCE_BRANCH_MODE=independent  # 또는 cumulative
export STEP_PREFERENCE_TARGET_SCOPE=immediate_text
export STEP_VISUAL_ONLY_CONTEXT_PREFERENCE=true
export STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES=false
export STEP_PREFERENCE_BLOCK_ORIGINAL_IMAGE=false
export STEP_PREFERENCE_WEIGHT=10
export STEP_PREFERENCE_BETA=0.1
export STEP_PREFERENCE_MARGIN=0
export STEP_ENTROPY_WEIGHT=0
```

Python CLI는 `--step_preference_prefix_mode v_off_insertion`과 `--step_preference_branch_mode independent|cumulative`다. 나머지 Stage2B 초기화 및 다중 노드 실행 설정은 기존 정책을 따른다. 위 loss 계수는 비교 실험 설정이며 전역 기본값을 바꾸지 않는다. 이 문서는 학습 실행 승인이 아니다.

이 모드는 Stage2B context preference, V-only, START/END 유지, shared-prefix 실행과 microbatch1을 요구한다. legacy packing과 `block_original_image=true`는 오류이며, 기존 prefix mode에서 cumulative를 지정해도 오류다. `until_next_latent`의 기존 target 매핑도 보존한다. dense 또는 CUDA Flex backend를 선택할 수 있고, Flex는 학습 attention dropout0이 필요하다. 지원하지 않는 조합은 명시적으로 실패한다.

CE는 원래 clean 실행을 유지하고 preference는 실제 `stats/logp_evidence_on/off`를 사용한다. `stats/logp_clean`은 이 모드에서도 evidence-on의 호환 별칭이다. tiny FP32/dropout0 모델의 전체 시퀀스 재계산 대비 latent/logit/score/gradient 검증과 nonreentrant checkpoint 검증은 방법의 수치 확인이며, 실제 학습의 메모리 적합성·수렴·정답률 개선을 뜻하지 않는다.

## 같은 이미지 evidence replacement

`evidence_replacement`는 각 reasoning step의 원본 full-prefix 점수와, 같은 이미지에서 고른 대체 evidence를 넣은 full-prefix 점수를 비교한다. 첫 step은 같은 이미지 encoder feature에서 latent 길이만큼 선택한다. 이미지 slot 수가 latent 길이 이상이면 without replacement, 더 적으면 with replacement이며 선택 위치는 정렬한다. 이후 step은 더 이른 검증된 latent block 하나를 donor로 쓴다. 대체 복사본은 detach되지만 원본 occurrence는 미분 가능하다. 선택은 training `SEED`, canonical sample ID, step ID와 고정 버전 `ilvr-evidence-replacement-v1`로 결정되므로 별도 replacement seed 옵션은 없다.

이 모드는 Stage2B context preference, V-only, 원본 이미지 접근, 독립 branch, entropy weight 0, 그리고 START/END 유지가 필요하다. `STEP_PREFERENCE_BLOCK_ORIGINAL_IMAGE=true`, cumulative branch, boundary suppression, legacy DPO와 결합하면 launch가 실패한다. `immediate_text`와 `until_next_latent` target scope는 모두 기존 정의대로 지원한다. clean CE 경로는 그대로이며 preference는 step 평균

$$
\mathcal L=\mathcal L_{CE}+\lambda_{pref}\left[-\frac1K\sum_k\log\sigma\left(\beta(s_{original,k}-s_{replacement,k}-m)\right)\right].
$$

로그의 `stats/logp_visual_original`, `stats/logp_visual_replacement`, `stats/replacement_margin`을 비교 점수로 사용한다. `stats/replacement_{donor,image}_margin_sum`과 `_count`는 source별 raw 합계/표본 수이며, source mean은 모든 rank의 raw 합과 count를 reduce한 뒤 계산한다. 따라서 batch별 mean의 평균으로 source 진단을 해석하지 않는다. 저장 model config에는 mode, training seed, sampling version 및 고정 selection/gradient semantics가 기록된다. 이 identity는 새 public Stage2B resume 경로를 추가하지 않는다.

다음은 실행 예시이며 여기서 학습을 시작하지 않는다. 두 노드 coordinator가 worker에 같은 mode, seed와 boundary 선택을 전달한다. `RUN_ID`는 고유해야 하며 checkpoint/log/result 경로는 현재 worktree의 표준 symlink 아래에 둔다.

```bash
export RUN_ID=evidence-replacement-egb8-s423-20260916
export STAGE1_CHECKPOINT=checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431
export STAGE2A_CHECKPOINT_PATH=checkpoints/${RUN_ID}-stage2a/checkpoint-211
export OUTPUT_DIR=checkpoints/${RUN_ID}
export LOG_FILE=logs/${RUN_ID}/train.log
export RESULT_DIR=results/${RUN_ID}
export SEED=42
export STEP_PREFERENCE_ENABLED=true
export STEP_PREFERENCE_MODE=stage2b
export STEP_PREFERENCE_OBJECTIVE=context_preference
export STEP_PREFERENCE_PREFIX_MODE=evidence_replacement
export STEP_PREFERENCE_BRANCH_MODE=independent
export STEP_PREFERENCE_TARGET_SCOPE=immediate_text
export STEP_VISUAL_ONLY_CONTEXT_PREFERENCE=true
export STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES=false
export STEP_PREFERENCE_BLOCK_ORIGINAL_IMAGE=false
export STEP_PREFERENCE_WEIGHT=10
export STEP_PREFERENCE_BETA=0.1
export STEP_PREFERENCE_MARGIN=0
export STEP_ENTROPY_WEIGHT=0
export STEP_SIDECAR_ATTENTION_BACKEND=dense
./run_multinode_stage2_training.sh --run-id "$RUN_ID" --tuning-method dora --zero-stage 2
```
