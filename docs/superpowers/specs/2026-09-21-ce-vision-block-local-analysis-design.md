# CE-only Stage 2: Vision block의 네 가지 국소 개입

상태: **design commitment v1 — 문서 승인 대기**. 대화에서 승인된 실험 정의를 구체화하고 설계 검토를 반영했다. 이 문서의 최종 승인과 구현 계획 승인은 아직 받지 않았다. 코드 구현·GPU 실행 결과가 아니다. 검토 기록은 [workflow evidence](2026-09-21-ce-vision-block-local-analysis-workflow.md)에 있다.

## 1. 목적과 승인된 범위

understood as: CE-only Stage 2의 동일한 가중치와 기준 풀이에서 한 Vision block의 Body 및 경계 접근을 바꿨을 때, 바로 다음 텍스트 구간의 토큰 예측과 정답 NLL이 얼마나 달라지는지 TEST 약 100개로 비교한다.

사용자는 CE-only baseline을 선택했고, 다음 네 조건을 순서대로 분석하도록 지정했다. 이어서 네 조건의 길이·position 유지, 각 site의 독립 분기, END 입력의 재계산, boundary-only에서 clean Body 고정이라는 대화 설계에 `ok`로 답했다. 이 승인은 설계 문서 작성까지이며 구현 승인으로 확대하지 않는다.

| 순서 / 이름 | 표기 | clean prefix에서 직접 읽지 못할 위치 | END 자리의 새 입력 |
|---|---|---|---|
| 1 / clean | START / Body / END | 없음 | 원래 END embedding |
| 2 / body_mask | START / X / END | 현재 Body 8개 | 원래 END embedding |
| 3 / boundary_mask | X / Body / X | 현재 START 및 원래 clean END | zero vector |
| 4 / full_mask | X / X / X | 현재 START, Body 및 원래 clean END | zero vector |

네 조건 모두 동일한 논리적 길이, target token IDs, 3축 position IDs를 유지한다. `X`는 토큰 배열 삭제를 뜻하지 않는다. 표의 K/V 접근 차단과 END embedding 대체라는 두 연산으로 정의한다. 학습, 물리 삭제, Body 재생성, 자유 생성, KL 추가는 이번 범위에 없다.

## 2. 모델과 실행 기반

- 학습 run: `stage2-ce-matched-gb16-s212-20260919-155759`, checkpoint-212.
- 준비된 adapter: `/root/workspace/ILVR/.worktrees/voff-reproduction-20260917/checkpoints/evaluation-staging/stage2-ce-matched-gb16-s212-20260919-155759/prepared-checkpoint-212`.
- 원본 base: `/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct`.
- 초기 Stage 1: reproduced checkpoint-431. Stage 1 3-epoch에서 시작한 다른 CE run으로 자동 교체하지 않는다.
- 기록된 정상 TEST400 정확도는 212/400=53%이며 이번 국소 지표와 구분한다.
- 작업 기반은 `.worktrees/voff-reproduction-20260917`의 검토된 코드다. main의 비슷한 옵션만으로 동등성을 가정하지 않는다. 구현 전 기존 변경과 리뷰 증거를 확인하고 별도 작업 공간에서 독립 구현한다.
- 해당 repository의 `venv/ilvr-venv`를 사용한다. `eval()`/gradient off, BF16 모델, FP32 log-softmax, decoder SDPA, vision FA2, 동일한 suffix attention backend를 사용한다. batch는 sample 1이다. attention mask가 달라지는 것을 제외하고 조건 간 backend·dtype·chunk 정책을 고정한다.
- 본 작업은 평가다. 학습용 두 노드/네 rank 정책을 평가의 필수 조건으로 확장하지 않는다. 자원 확인 후 가용 GPU 하나에서 순차 처리하는 최소 경로를 우선한다.

모델, tokenizer, 실행 코드와 유효 설정의 identity는 실제 launch 때 기록한다. 이번 문서는 가중치 전체의 hash를 새로 계산하거나 GPU를 점유하지 않는다.

## 3. 100개 샘플과 평가 위치

데이터는 `data/TEST.jsonl`이다. 조사 시 SHA-256은 `6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`이다.

기존 main100의 `selection.requested_indices`를 그대로 재사용한다. 원본 파일은 `/root/workspace/ILVR/results/server-comparison-20260912/analysis_completed/visual-causal-panel-20260910t115000z/ce/no_latent_execution/metadata.json`이며 조사 시 SHA-256은 `fb1f15fef5339abde6ba74d140e5f9a1f8823cedafe8aef1d4bb245a8e0286aa`이다. 0-based 100개 고유 index이며 creation/deletion/update/selection 구성은 25/30/15/30이다. seed만으로 재추출하지 않는다.

원본 `sequence_plan`을 직접 세었을 때, 100개 모두에 site가 있으며 latent 바로 다음의 비어 있지 않은 text 항목은 총 327개다. 이는 **구조상 후보 수**이며 tokenizer 적용 후 최종 유효 site 수를 뜻하지 않는다. 306 또는 11,811을 목표 분모로 맞추지 않는다.

- site는 `(source_index, plan의 latent 항목 index)`로 식별한다. 한 site의 target은 latent 바로 다음 text 항목 하나(`immediate_text`)다. 별도의 후속 final-answer text 항목까지 합치지 않는다.
- 기준 텍스트는 TEST의 `sequence_plan` annotation이다. 모델이 만든 clean 문장을 정답으로 사용하지 않는다.
- 기존 `concatenated-v1` 렌더링, processor, latent-size 8, 검증된 offset mapping을 재사용한다. 보조 이미지 경로는 구조 메타데이터로만 쓰며 helper image/pixels를 모델에 넣지 않는다. 원본 입력 이미지만 사용한다.
- START/Body/END, padding, chat control token은 평가 target에서 제외한다. 인접 text/text 경계를 가로지르는 토큰은 기존 승인된 규칙으로 평가 집합에서만 제외하며 입력 텍스트/teacher-forced prefix에는 그대로 둔다.
- 실제 첫 text token을 정확히 매핑하지 못한 site, 빈 token target, 잘못된 latent 구조는 네 조건에서 공통 제외하고 이유를 기록한다. 첫 토큰 지표를 임의의 나중 토큰으로 대체하지 않는다.
- `original_final_answer`가 비어 있다는 이유만으로 site를 제외하지 않는다. 이 평가의 target은 유효한 기준 text 구간이다. annotation의 진실성 전체를 별도로 검증한 평가라고 주장하지 않는다.
- 결정론적 입력 검증 후 공통 site/target 집합을 고정한다. 추론 실패나 OOM으로 특정 조건의 분모만 줄이지 않는다. 실행 결과는 incomplete로 표시하고 원인을 해결한 뒤 동일 집합을 평가한다.

런타임에서 생성할 선택/토큰 manifest는 작업 repository의 `data/`에, 지표와 제외 사유·hash 기록은 `results/<run-id>/`에 둔다.

## 4. 한 site의 정확한 실행 의미

`P · START · B1 … B8 · END · y1 … ym`에서 P는 원본 이미지, 질문과 그 site보다 앞선 clean history다. `e`는 END의 원래 논리적 position이다.

1. 기존 Stage 2 clean rollout으로 latent Body를 계산한다. 앞선 텍스트는 annotation을 teacher forcing하며, 미래 target은 causal mask를 통해 과거에 영향을 주지 못한다.
2. clean prefix의 layer별 K/V 중 END **이전**만 branch의 참조 범위로 허용한다. clean END 및 clean target의 K/V는 어느 조건에서도 재사용하지 않는다.
3. suffix 입력은 `[END_input, y1, ..., y(m-1)]`이고 predictor의 원본 token 열 index는 `[e, e+1, ..., e+m-1]`이다. **position IDs는 그 열에서 원래 3축 값을 그대로 선택한다.** 이 index 표기를 scalar position 재생성 또는 mRoPE 재계산으로 구현하지 않는다. 평가에서 제외한 crossing token이 있다면 전체 원래 suffix는 유지하고 해당 score row만 제외한다.
4. 모든 decoder layer에서 표의 root K/V 위치를 차단한다. suffix는 자기 branch의 causal 과거만 볼 수 있으며, 다른 site/조건의 suffix에는 접근하지 못한다.
5. boundary_mask/full_mask는 END_input만 zero vector로 대체한다. 나머지 GT text embedding은 동일하다. 새 END query는 허용된 문맥과 자기 자리를 사용해 계산하며, 이후 suffix는 이 **재계산된** END 상태를 참조할 수 있다. 이를 END 계산 자체의 제거라고 부르지 않는다.
6. boundary_mask의 Body 입력·clean K/V는 clean에서 생성한 그대로다. START가 Body 생성에 이미 준 영향은 남는다. 따라서 경계의 직접 접근/입력 효과를 측정하며 START의 모든 간접 효과를 제거하는 실험은 아니다.
7. full_mask는 현재 clean block으로부터의 직접 접근을 막지만 원본 이미지, 이전 text/latent와 GT text prefix를 유지한다. 보고서에서 '모든 시각 정보 제거'라고 해석하지 않는다.

각 site는 동일한 clean history에서 시작한다. 앞 site의 ablation을 뒤 site로 누적하지 않는다. 네 조건은 sample마다 1→2→3→4 순으로 실행하고, 표 역시 이 순서로 작성한다. 100개 전체의 clean 캐시를 보관하는 새 시스템은 만들지 않는다.

논리적 sequence 길이 유지와 실제 suffix replay의 tensor row 수를 구분한다. root+suffix로 캐시를 참조하는 실행이더라도 원본 token 배열에서 block을 삭제하거나 position을 이동하지 않는다.

## 5. 지표 계약

S는 공통 유효 site 수, n_s는 site s의 유효 target token 수, N=Σ_s n_s다. `p_c(s,j)`는 조건 c에서 동일한 정답 prefix를 teacher forcing한 vocabulary 확률이고 `a_c(s,j)=argmax p_c(s,j)`다. 동률은 고정 vocabulary 순서에서 가장 작은 token ID로 해소한다. logits processor, repetition penalty, sampling은 적용하지 않는다.

| 지표 | 정의 |
|---|---|
| 첫 토큰 변경 | Σ_s 1[a_c(s,1) ≠ a_clean(s,1)] / S |
| 구간 내 변경 | Σ_s 1[어느 j에서 a_c(s,j) ≠ a_clean(s,j)] / S |
| 전체 토큰 변경 | Σ_s,j 1[a_c(s,j) ≠ a_clean(s,j)] / N |
| 평균 NLL | −Σ_s,j ln p_c(s,j)[y_s,j] / N |
| ΔNLL | NLL_c − NLL_clean |

NLL은 자연로그, 토큰 가중 평균이다. sample별/구간별 평균을 다시 단순 평균하지 않는다. FP32 log-softmax에서 정답 log-prob를 얻고 합계는 충분한 정밀도로 누적한다. clean의 변경 수와 ΔNLL은 0이어야 한다. ΔNLL은 음수도 허용하며 clipping하지 않는다.

주 결과는 4행×5개 지표다. 변경 지표는 `분자/분모 (백분율)`로 표시한다. site별 결과와 token별 GT ID, 조건별 argmax ID·정답 NLL, 원래 위치도 저장해 집계를 재현할 수 있게 한다. 전체 vocabulary logits를 지속 저장하지 않는다. 다른 모델/서로 다른 평가 경로의 clean을 빼지 않는다.

이 결과는 기준 풀이 위의 국소 예측 민감도다. 자유 생성된 문장의 변경률, 최종 정답 정확도, 의미적 정답성 또는 인과적 시각 이해의 증명으로 확대하지 않는다.

## 6. 구현 접근과 대안

권장: 기존 clean-prefix suffix replay를 기반으로 별도 분석 entrypoint와 offline 집계 함수를 추가한다. 현재 sidecar는 연속된 차단 범위 하나와 `zero_first_embedding`을 지원하지만, boundary-only의 검증 계약은 그대로 지원하지 않는다. 전용 분석 planner에서 START만 root 차단하고 clean END는 공통 prefix cutoff로 제외하며 END query를 zeroing하는 경로가 필요하다. 현재 training validator를 우회하거나 기존 설정의 의미를 바꾸지 않는다.

기존 `score_selected_hidden`은 정답 mean log-prob/entropy 중심으로 반환하므로, 분석 경로에는 선택한 위치의 argmax와 개별 정답 NLL을 반환하는 작은 scorer가 필요하다. 기존 training return contract는 유지한다. 실제 파일 분할과 테스트 명령은 문서 승인 뒤 구현 계획에서 확정한다.

대안은 site마다 전체 입력을 재계산하는 방식이다. 독립적인 비교 oracle로는 유용하지만, Body까지 intervention 아래에서 재계산하면 승인된 clean Body 고정 조건을 바꾸므로 본실험의 기본 경로로 선택하지 않는다. cached replay의 독립 oracle은 clean prefix 값과 개입 지점을 그대로 고정하는 동등한 layer별 reference여야 한다.

구체적으로 tiny reference는 sample/site 한 개씩 실행하고, 정수 START/Body/END 위치에서 dense visibility를 별도로 구성한다. production planner, production visibility builder, branch packing을 호출하지 않는다. 모델 가중치와 검증된 decoder 연산은 공유할 수 있으므로 이것은 decoder 수학 전체의 독립 구현 검증은 아니다. '정상 clean'은 동일 Stage 2 모델의 `forward → _forward_stage2`에서 `stage2_shared_prefix_inputs=None`으로 실행한 원래 clean 경로다. 분석 전용 branch를 생성한 뒤 그것을 정상 clean이라고 부르지 않는다. 구현 계획에서 실제 capture·reference 진입점과 공유하는 함수 목록을 고정한다.

## 7. 검증과 실행 gate

먼저 실패하는 작은 causal regression test를 작성하고 구현한다. 검증은 다음 세 종류를 한 번의 집중된 세트로 수행한다.

- **지표/정렬:** 손으로 계산 가능한 고정 logits와 여러 길이의 target으로 S/N, 첫 predictor의 one-token shift, any-change, token weighting, negative ΔNLL, special/crossing token 제외를 검증한다. 집계는 모델을 로드하지 않고 저장 row에서 재계산 가능해야 한다.
- **경로:** 한 site와 두 site의 tiny model에서 실제 layer별 prefix/attention witness를 확인한다. clean 원본 END K/V를 변조해도 replay 출력에 영향이 없어야 한다. 차단된 root K/V를 capture 후 변조해도 해당 조건 출력은 불변이어야 한다. visible Body가 남는 boundary_mask는 그 Body를 변조하면 반응할 수 있어야 한다. clean Body를 변조하면서 간접 경로 불변성을 요구하지 않는다. 새 END query zeroing과 self/causal suffix 접근을 별도로 확인한다.
- **실제 모델 최소 확인:** 구현·독립 리뷰 후 고정 main100 중 처음 나오는 1-site sample과 처음 나오는 multi-site sample을 대상으로 실제 CE artifact를 로드해 네 조건을 실행한다. 모든 조건의 IDs/position/target 길이, clean replay와 정상 clean의 selected predictions/NLL, 반복 clean의 재현성, dtype/backend, nonfinite 여부를 기록한다.

smoke 샘플은 tokenization 후 유효 site 수로 정하며 source index 오름차순에서 고른다. 정확히 1-site인 sample이 없으면 첫 multi-site sample의 첫 유효 site만 **평가 대상으로 선택**해 단일 branch를 검사하고, 같은 sample의 전체 유효 site로 복수 branch를 검사한다. 이 fallback은 원본 prefix나 Body를 잘라내지 않는다. multi-site sample 자체가 없다면 실제 모델의 복수-site witness가 미충족이므로 본실험 시작을 보류하고 설계를 다시 확인한다.

실제 모델 확인에서 clean replay의 argmax가 정상 clean과 다르면 차단 효과로 해석하기 전에 원인을 해결한다. 실행 전 gate로 BF16 clean 비교의 token NLL 절대차는 최대 0.01 nat, 평균 0.001 nat 이하를 요구한다. 이는 결과를 보기 전에 정한 허용 기준이며 실행 동등성의 증명은 아니다. tiny FP32 oracle은 logits에 `atol=1e-5, rtol=1e-5`를 적용한다. 반복 실행의 clean argmax 변경은 0이어야 한다. GPU 결과를 본 뒤 기준을 임의로 완화하지 않는다. 같은 경로의 no-op와 단순한 자기 비교는 attention 구현 자체를 독립 검증한 것으로 인정하지 않는다.

경로·정렬·clean 대조가 통과하기 전 100개 본실험을 시작하지 않는다. 검증 실패 시 결과를 보존하고 가장 작은 수정과 causal regression으로 해결한다. retry/recovery framework를 추가하지 않는다.

실행은 자원 충돌/입력/프로토콜을 점검한 뒤 고유 run-id로 detached 실행한다. output은 `results/<run-id>/`, log는 `logs/<run-id>/`, dataset manifest는 `data/`에 둔다. 평가에 새 모델 준비가 필요하다면 `checkpoints/evaluation-staging/<run-id>/`만 사용한다. 기존 checkpoint나 다른 작업의 결과를 덮어쓰지 않는다.

## 8. 근거와 다음 단계

현재 확인한 근거는 CE launch/summary/adapter metadata, TEST의 구조, 기존 `src/step_preference/sidecar.py`, `spans.py`, `src/stage2/data.py`, `stage2_shared_prefix.py` 소스다. 새 evaluator의 실행 동등성은 아직 검증되지 않았다.

설계 critique와 cold read를 기록하고 최종 design commitment를 이 문서와 일치시킨 뒤 사용자에게 문서 승인을 요청한다. 승인 후 `writing-plans`로 구현 계획을 작성하고 계획 승인 후 implementer와 독립 reviewer를 통해 구현한다. 이 문서 작성 완료는 분석 코드 준비 또는 100개 실험 완료를 뜻하지 않는다.

## 9. Design commitment v1

- **합의한 문제/범위:** 위 CE-only artifact, 고정 main100, immediate-text teacher forcing, 독립 site, 길이/position 유지, 4조건과 5개 지표. boundary-only의 Body는 clean에 고정한다.
- **최종 설계:** §§2–7을 그대로 적용한다. 기존 clean-prefix replay를 기반으로 분석 경로를 추가하고 annotation target에 대한 예측/NLL을 저장·집계한다. 실제 런타임 검증을 통과한 뒤 본실험을 허용한다.
- **proposal v1에서의 변경:** cold read 후 scalar처럼 보이던 position 표기를 원본 3축 열 선택으로 명확히 했다. 독립 oracle과 정상 clean 진입점, smoke 샘플 fallback을 구체화했다. 실험 대상/개입/지표/해석은 바꾸지 않았다.
- **first nail / 상태:** tiny model의 차단된 captured K/V 변조 불변성과 visible-Body 양성 대조, 독립 reference 비교, 이어서 실제 CE 두 smoke 사례의 clean fidelity. **아직 실행하지 않았다.** 현재 승인은 설계 작성까지이고 필요한 분석 코드도 없기 때문이다. §7의 runtime gate를 통과하기 전에는 100개를 실행하지 않는다.
- **검토 차이:** 지표/독립 target 정의는 문서 수준에서 수용 가능하지만, 실제 개입 경로의 타당성은 runtime witness 전까지 미확인이다. 이 차이를 리뷰 합의로 해소했다고 주장하지 않는다. cold read는 minor gaps였고 위 세 설명을 보완했다.
- **남은 미검증 전제:** 선택 모델에서의 replay fidelity, 경계-only의 실제 차단 경로, 최종 tokenized S/N, 자원/메모리 가용성. runtime gate에서 확인한다. 원본 이미지·clean Body의 간접 영향 및 annotation 기반 해석 한계는 §4–5처럼 유지한다.
- **위험 면제:** 없음. 문서 승인이 runtime gate를 면제하지 않는다. 이 commitment의 문서 승인은 구현 계획 작성만 허용하며, 이후 계획 검토·구현 위임·독립 리뷰·검증 gate를 유지한다.
