# `v_off_insertion` 구현 설명

## 목적과 범위

이 문서는 Stage 2B의 **visual-only context preference** 경로인 `v_off_insertion`을 설명한다.
입력 구성, 점수와 loss, gradient 연결 및 체크포인팅을 새로운 환경에서도 참고할 수 있도록 정리한다.
아래 설명은 구현의 정적 호출·데이터 흐름을 추적한 것이며, 학습 결과나 성능 검증을 뜻하지 않는다.
옵션의 일반 기본값, insertion에 필요한 값, 참고 실험 설정을 구별한다.

## OFF와 ON이 만드는 입력

입력 계획 생성기 `build_insertion_plan`은 원래 입력 위치에서 새 입력 위치로 가는 매핑을 만든다.
원래 이미지 블록은 `vision_start + image tokens + vision_end`,
latent 블록은 `latent_start + latent_pad × K + latent_end` 구조여야 한다.
여기서 `S`, `E`는 latent 경계 토큰, `I`는 원래 이미지 블록 전체,
`L_i`는 i번째 블록의 K개 연속 latent 상태, `T_i`는 해당 target text를 뜻한다.
여러 원본 이미지가 있으면 `I`는 원래 순서의 **모든** 이미지 블록을 뜻한다.

| 경로 | 입력 구성과 목적 |
|---|---|
| clean CE | 원래 위치의 이미지와 모든 latent 블록을 사용하는 기존 Stage 2 입력. CE를 계산한다. |
| OFF | 모든 원본 이미지 블록을 경계까지 삭제하고, 모든 latent 내부 pad를 삭제한다. 각 `S E`는 남긴다. |
| ON | OFF를 바탕으로 선택된 `S` 직전에 `I`를 삽입하고 그 블록의 latent pad를 복원한 뒤, latent 상태를 새로 계산한다. |

OFF는 latent 값을 0으로 바꾸거나 attention mask만 씌우는 경로가 아니다.
삭제 후 남은 token ID를 `embed_tokens`로 다시 임베딩하므로 clean latent/decoder hidden을 재사용하지 않는다.
ON의 삽입 이미지 feature와 텍스트 embedding은 clean rollout **이전**의 `base_inputs_embeds`에서 가져온다.
이미 계산된 clean latent를 ON에 복사하지 않는다.

예를 들어 두 reasoning step이 있는 입력의 의미상 구성은 다음과 같다.
`P`는 이미지 블록 외에 보존되는 prompt 및 이전 텍스트이며, 실제 원본 이미지 위치는 입력에 따른다.

```text
clean       : I P S L1 E T1 S L2 E T2
OFF         :   P S    E T1 S    E T2
independent ON1: P I S L1 E T1 S    E T2  → T1 점수
independent ON2: P   S    E T1 I S L2 E T2 → T2 점수
cumulative ON : P I S L1 E T1 I S L2 E T2 → T1, T2 점수
```

여기서 선택 step은 보조 비교 경로(sidecar)를 구성하는 계획에 전달되어 검증된 모든 reasoning span의 visual branch를 뜻하며, insertion scorer가 별도로 sampling하지 않는다.
`independent`는 선택된 step마다 별도 ON branch를 만든다.
ON2의 이전 문맥은 OFF이므로 ON1에서 생성한 latent를 이어받지 않는다.
`cumulative`는 모든 선택 step을 하나의 ON branch에 넣는다.
뒤 step은 앞서 삽입한 이미지와 재계산한 latent를 causal 문맥으로 사용할 수 있다.
각 step의 OFF 비교 점수는 동일한 전체 OFF 실행에서 얻는다.

## KV, 위치, latent의 흐름

삽입 실행기 `score_v_off_insertion`의 처리 순서는 다음과 같다.

1. OFF token ID로 `get_rope_index`를 호출하여 이미지 없는 위치 ID를 새로 계산한다.
2. OFF decoder를 `stage2_capture_kv=True`, `use_cache=False`로 실행한다.
   각 layer의 KV와 OFF hidden을 얻고 target predictor 행에서 OFF 점수를 계산한다.
3. 각 ON branch에서 첫 삽입 위치 **직전까지의 OFF KV**만 잘라 공유한다.
   `independent`는 해당 step 직전, `cumulative`는 첫 선택 step 직전이 분기점이다.
4. 삽입 후의 전체 ON token ID와 반복된 image grid로 multimodal RoPE 위치를 다시 계산한다.
   debugging이 켜져 있으면 삽입 전 ON/OFF 위치가 같은지 검사한다.
   ON suffix에는 새 위치 ID를, 공유 prefix에는 실제 OFF 실행의 KV를 사용한다.
5. ON suffix embedding과 prefix KV를 decoder에 넣고 선택된 latent 블록을 rollout한다.
   그 결과로 최종 ON suffix를 실행하여 선택 target의 점수를 계산한다.

latent rollout은 각 latent slot 직전까지 decoder를 실행하고, 마지막 hidden state를 그 slot의 embedding으로 넣는다.
앞서 생성한 latent는 다음 slot 계산에 포함된다. latent는 이산 token을 sampling한 결과가 아니다.
이 경로는 `use_cache=False`의 연속 recurrence이며, 공유 KV는 별도 `stage2_prefix_key_values` 인자다.
ON과 OFF의 점수 및 공유 prefix KV를 detach하지 않아 preference gradient가 연결된다.
실제 parameter 갱신 대상은 DoRA 등 선택한 tuning/freeze 설정에 따른다.

ON의 attention은 prefix와 suffix를 잇는 causal mask를 사용한다.
Flex 경로는 설정된 sliding window도 적용한다.
삽입 이미지에 대한 별도 text-query 차단은 없으므로, causal/window 조건 안의 ON 텍스트는 이미지를 직접 볼 수 있다.
따라서 ON/OFF 차이를 **latent만을 통한 시각 정보 전달 효과**로 해석할 수는 없다.
`step_preference_block_original_image=false`는 OFF에 이미지를 남긴다는 뜻도 아니다.
insertion에서는 OFF 이미지를 구조적으로 삭제하며 이 옵션의 `true`는 거부한다.

## 동일 target과 loss

target 범위를 선택하는 `immediate_text`는 sequence plan에서 latent 바로 다음에 오는 하나의 text 항목을 target으로 삼는다.
다음 latent 전까지의 모든 text를 뜻하는 `until_next_latent`와 구별한다.
입력 길이와 위치는 바뀌어도 OFF/ON의 target token ID와 순서는 같다.
계획 생성기는 원래 target ID를 검사하고, 새 target 위치 바로 앞의 predictor 위치를 매핑한다.
두 경로 모두 같은 정답 텍스트를 teacher forcing한다. 서로 다른 답변을 생성해 선호쌍으로 쓰지 않는다.

step i에서 실제 선택된 `target_token_ids`의 개수를 n_i라 하면 점수와 preference loss는 다음과 같다.

```text
s_ON,i  = (1/n_i) Σ_j log pθ(y_i,j | ON 문맥, 앞선 정답 token)
s_OFF,i = (1/n_i) Σ_j log pθ(y_i,j | OFF 문맥, 앞선 정답 token)
L_pref  = mean_i [−log sigmoid(β × (s_ON,i − s_OFF,i − margin))]
L_total = L_clean_CE + λ_pref × L_pref + λ_entropy × L_entropy
```

점수 함수 `score_selected_hidden`은 선택 predictor만 LM head에 투영하고 float32 log-softmax를 계산한다.
token log-probability를 step 안에서 평균한 뒤, `context_preference_loss`가 step별 loss를 평균한다.
긴 step에 token 수만큼 가중하는 전체 합산은 아니다.

학습을 관리하는 Trainer는 `visual_on`을 preference의 양쪽 중 ON 점수로 사용한다.
이 경로의 `stats/logp_clean`은 이름과 달리 ON 점수이며 `stats/logp_evidence_on`에도 기록된다.
`outputs.loss`에서 가져오는 clean CE는 원래 입력의 별도 Stage 2 rollout/forward에서 계산한다.
CE supervision은 labels가 `-100`이 아닌 위치에 적용되므로 preference target 범위와 같다고 가정하면 안 된다.
이 context preference 경로에는 고정 reference model의 log-ratio나 negative artifact 점수가 들어가지 않는다.
reference 점수 차이를 차감하는 별도 `legacy_dpo` objective와 구별해야 한다.

entropy 항은 같은 target 위치에서 `relu(entropy_margin − (H_OFF − H_ON))`을
token별로 계산하고 step별 평균 후 step 사이를 평균한다.
H는 각 target의 predictor 위치에서 전체 vocabulary 분포에 대해 계산한 entropy `−Σ_v p(v) log p(v)`이며, 정답 token 하나의 surprisal은 아니다.
현재 코드에서 insertion이고 `λ_entropy=0`이면 total에 넣는 entropy를 detach하여 그 항의 backward 연결을 끊는다.
entropy forward 계산과 로그 metric은 남는다.
또한 insertion 학습 호출이 `return_outputs=False`이면 쓰이지 않는 clean preference용 selected logits를 요청하지 않는다.
clean CE용 logits, ON/OFF 점수 계산은 유지된다. 이 변경만으로 속도나 수치적 동등성이 입증되지는 않는다.

## 옵션: 일반 기본값과 insertion 설정

일반 기본값은 설정 객체와 CLI가 사용하는 값이다.
참고 실험 설정은 insertion 실행에서 선택했던 값이며, 새로운 환경의 자동 기본값이나 필수 조건을 뜻하지 않는다.

| 옵션 | 일반 기본값 | insertion 조건 / 참고 실험 설정 |
|---|---|---|
| `step_preference_enabled`, `step_preference_mode` | `false`, `stage2a` | `true`, `stage2b` |
| `step_preference_objective` | `context_preference` | 동일; legacy DPO 불가 |
| `step_preference_prefix_mode` | `clean_prefix` | `v_off_insertion` 지정 |
| `step_shared_prefix_enabled` | `false` | `true` 필수 |
| `step_visual_only_context_preference` | 자동; 활성 context preference에서 `true` | `true` 필수 |
| `step_visual_off_suppress_latent_boundaries` | 자동; V-only context preference에서 `true` | **`false`를 명시**하여 S/E 보존 |
| `step_preference_block_original_image` | `false` | `false` 필수; OFF 이미지 삭제와 별개 |
| `step_preference_branch_mode` | `independent` | `independent` 또는 `cumulative`; 참고 설정은 전자 |
| `step_preference_target_scope` | `immediate_text` | 참고 설정도 동일 |
| `ce_weight`, `step_preference_weight` | `1`, `1` | CE는 `1` 필수; 참고 설정도 둘 다 `1` |
| `--step_preference_beta`, `step_preference_margin` | `0.1`, `0` | 참고 설정도 동일; β는 기존 `--step_dpo_beta`도 허용하며 두 CLI의 저장 필드는 `step_dpo_beta`, 환경변수는 `STEP_PREFERENCE_BETA` |
| `step_entropy_weight`, `step_entropy_margin` | `0.1`, `0.1` | 참고 설정은 **`0`, `0.1`** |
| `step_sidecar_attention_backend` | `flex` | `dense`/`flex`; 참고 설정은 Flex, clean SDPA |
| `step_on_layer_checkpoint_stride` | `1` | `1`/`2`/`4`; 참고 설정은 `1` |

insertion은 microbatch 1, canonical image/latent 블록과 일치하는 image grid 및 실제 image 입력을 요구한다.
legacy packing/negative artifact는 이 경로에서 사용하지 않는다.

## 체크포인팅과 확인 범위

참고 실험 설정은 `GRADIENT_CHECKPOINTING=true`이며 CE/OFF/ON layer checkpointing을 유지한다.
Stage 2 초기화는 gradient checkpointing이 활성화되면 layer checkpoint를
`use_reentrant=False`로 설정하고, PEFT(LoRA/DoRA) + ZeRO-3 + layer checkpoint 조합은 거부한다.
nonreentrant 방식은 이 구현의 정책이고, 활성화 값 `true`는 참고 실험의 선택이다.
참고 실험의 분산 optimizer 설정은 ZeRO-2다.

ON에는 두 수준의 checkpoint가 있다. 삽입 실행기는 ON rollout과 최종 점수 계산 전체를
nonreentrant checkpoint로 감싸며 `preserve_rng_state=True`를 지정한다.
동시에 내부 decoder 호출에 `stage2_checkpoint_prefix=True`를 전달하여 prefix KV가 있어도
native layer checkpoint 경로를 선택할 수 있게 한다. OFF selected-score 계산에도 별도 checkpoint가 있다.
decoder는 training 및 gradient checkpointing 활성화 상태에서 ON의 `layer_index % stride == 0`인 layer를 checkpoint한다.
따라서 `--step_on_layer_checkpoint_stride 1/2/4`는 각각 모든/매 2번째/매 4번째 ON layer를 선택한다(0부터 시작).
stride는 CE/OFF layer checkpoint와 ON 전체 branch checkpoint를 끄지 않는다.

이 문서는 구현 동작과 참고 설정을 설명하며, stride별 성능·gradient·학습 결과의 동일성을 보장하지 않는다.
새 환경의 수치 재현성과 성능은 별도 실행 증거로 확인해야 한다.
