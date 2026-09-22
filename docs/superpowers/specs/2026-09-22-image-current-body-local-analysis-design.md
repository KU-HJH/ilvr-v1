# 원본 이미지 직접 접근 및 직전 visual Body 국소 차단

상태: **design commitment v1 — 서면 승인 대기**. 실험 정의에 대한 대화 승인(`ㅇㅋ`)을 받았으며, 이 문서는 서면 설계 검토용이다. 구현·새 GPU 실험은 아직 승인하거나 실행하지 않았다. 기존 sub-agent 구현 및 독립 리뷰 선택을 유지한다.

이 문서 승인은 구현 계획을 작성할 권한을 부여한다. 작성된 계획의 사용자 검토·승인 뒤에 위임 구현을 시작한다. 그 계획에는 여기 명시한 causal test, GPU smoke와 통과 후 본실험까지 포함하며, 승인된 범위 안에서 각 작업마다 승인을 다시 요구하지 않는다. 설계 변경이나 별도의 구현 전 GPU premise probe가 필요하면 해당 변경/실험을 별도로 제시한다. 모든 상대 경로는 아래 작업 폴더 기준이다.

## 1. 목적과 범위

understood as: beta=0.5 Stage 2 모델에서 원본 이미지를 보고 생성한 visual block과 이전 clean 문맥을 고정하고, 각 visual block 직후 텍스트를 예측할 때 원본 이미지에 대한 직접 attention 접근을 차단한다. 두 번째 조건은 여기에 평가 직전 Body만 추가 차단한다. 기존 다섯 지표와 동일한 100개 샘플로 측정한다.

사용자가 확정한 사항:

- visual block은 원본 이미지를 보고 생성한 clean block을 사용한다.
- 2번의 추가 차단 범위는 평가 직전 visual block의 Body뿐이다.
- 원본 이미지로 visual block을 계산하는 기존 방식은 유지한다.
- 마지막 대화의 clean / 이미지 차단 / 이미지+직전 Body 차단 표와, 이전 문맥·다른 block·START/END·길이·position 유지에 `ㅇㅋ`로 답했다.

이번 설계는 자유 생성, 재학습, 이미지 픽셀 대체, 토큰 삭제, 전체 visual block 차단, image-off 조건에서 latent 재생성을 포함하지 않는다. 이전에 제시한 architectural 경로를 유지한다.

## 2. 모델·코드·입력

- 작업 폴더: `/root/workspace/ILVR/.worktrees/ce-block-analysis-20260921`, 기존 격리 branch `feat/ce-block-analysis-20260921`.
- 구현 기준 revision: `fe3aadd066aa33e800f62b60993fe3c0e013fa07`. Canonical main의 Transformers와 동일하다고 가정하지 않는다.
- 모델: `checkpoints/evaluation-staging/voff-immediate-mask-b05-fixed-20260919-021026/prepared-checkpoint-212`.
- base: `checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct`.
- 학습 provenance: beta=0.5, preference weight=1.0, immediate_text, 212 updates. 이번 실행은 평가다.
- 데이터: `data/TEST.jsonl`, SHA256 `6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`.
- 표본 선택: `/root/workspace/ILVR/results/server-comparison-20260912/analysis_completed/visual-causal-panel-20260910t115000z/ce/no_latent_execution/metadata.json`, SHA256 `fb1f15fef5339abde6ba74d140e5f9a1f8823cedafe8aef1d4bb245a8e0286aa`의 기존 requested_indices 100개. 재추출하지 않는다.
- 비교 manifest: `data/ce-vision-block-analysis/b05-block-main100-v2-20260922-044315/manifest.json`. 각 입력 tensor hash와 source/plan/target position/target ID가 동일해야 한다. 기존 S=327, N=12,843, 제외=0. 불일치는 실행 중단 사유이며 분모를 강제로 맞추거나 실패 샘플을 제외하지 않는다.
- 이 manifest의 site별 `target_positions`/`target_ids` 목록을 평가 위치의 정확한 규범으로 삼는다. 이는 sequence_plan에서 현재 latent 직후의 immediate-text를 기존 offset mapper로 매핑한 목록이다. 새 prepare도 같은 mapper와 기존 special/crossing-token 제외를 사용해 이 목록과 대조하며, 사람이 텍스트 경계를 다시 추정하거나 다음 visual block 너머로 확장하지 않는다.
- 확인된 원본 데이터의 image_input 수는 선택된 100개 모두 1개다. tokenize 후에도 단일 원본 이미지 span과 image_grid_thw에 맞는 payload 위치를 검증한다.

## 3. 정확한 개입 정의

`P · START_i · Body_i · END_i · T_(i+1)`에서 `e`는 현재 END의 원래 위치다. `I`는 원본 prompt의 **이미지 payload** token 위치 집합으로, 실제 `input_ids == model.config.image_token_id`에서 얻고 원본 이미지 span 안인지 검증한다. `<vision_start>`/`<vision_end>`는 I에 포함하지 않는다. `B_i=[body_start, body_end)`는 현재 Body 8개이며 `body_end=e`다.

| 이름 | q ≥ e에서 추가로 차단할 key | END 입력 embedding |
|---|---|---|
| clean | 없음 | 원래 값 |
| image_mask | I | 원래 값 |
| image_body_mask | I ∪ B_i | 원래 값 |

모든 layer에 동일한 causal mask를 적용한다. `q < e`는 세 조건 모두 원래 causal visibility 그대로다. `q ≥ e`에서는 표의 key만 차단한다. START/END, 다른 Body와 이전 텍스트에 대한 기존 causal 접근은 유지한다. 각 site는 독립적으로 같은 clean 상태에서 분기한다. 앞선 site의 차단 결과를 다음 site로 누적하지 않는다.

1. 원본 이미지 접근을 허용한 기존 Stage 2 경로로 clean rollout을 한 번 수행한다. 정답 텍스트는 기존 annotation으로 teacher forcing한다.
2. 생성된 latent를 포함한 최종 decoder 입력 embedding과 3축 position, full-length clean K/V 및 점수를 관측한다.
3. 각 site/조건을 full-original-length로 재계산한다. **모든 입력 embedding을 그대로 유지**하며 END도 zeroing하지 않는다. 조건별 차이는 위 attention mask뿐이다.
4. 모든 layer의 `position < e` K/V가 clean과 bit-identical인지 검사한다. 이를 통해 visual Body와 앞선 문맥 계산이 바뀌지 않았음을 실행 중 확인한다. 보관한 K/V를 decoder 입력으로 재사용하는 short-suffix 방식으로 전환하지 않는다.
5. 기존 immediate-text target 위치만 점수화한다. 첫 정답 토큰의 predictor는 END 위치다. 뒤의 미평가 위치는 causal mask 때문에 평가 위치에 영향을 줄 수 없다.

이는 **원본 이미지 직접 접근의 국소 차단**이다. 이미지 정보가 이미 포함된 Body, 이전 hidden state, image boundary의 clean K/V, teacher-forced 텍스트 등 간접 경로는 유지된다. image_body_mask도 전체 시각 정보 제거 조건이 아니다.

## 4. 실행 구조와 호환성

기존 prepare → smoke → run → summarize 경로를 확장한다. CLI에는 명시적인 `--analysis-profile image-local-v1`을 추가하고 세 조건을 선택하게 한다. 기본값은 기존 네 조건의 `block-local-v2`로 유지하며 알 수 없는 profile은 실패시킨다. 기존 CLI 호출과 기존 결과의 offline summarize는 계속 동작해야 한다.

기존 네 조건의 계약은 `clean`, `body_mask`(현재 Body key 차단), `boundary_mask`(현재 START key 차단 및 END input zero), `full_mask`(둘의 합집합)이며 모두 현재 END 이후 query에 적용한다. 기존 진입점은 `scripts/analyze_vision_block.py`, argparse 계약은 `src/evaluation/vision_block_analysis.py::_parser`다. 새 profile은 같은 네 subcommand 모두에서 명시적으로 선택하고 새 manifest와 실행 identity에 기록한다. 명시한 profile과 저장된 profile이 다르면 실패한다. profile 필드가 없는 기존 manifest/결과는 기존 `block-local-v2`로만 해석한다. 새 prepare manifest는 profile과 이미지 위치 정보가 추가되어도 원본 입력·평가 위치 규범은 섹션 2와 같아야 한다.

- `vision_block_inputs.py`: 기존 입력과 target 매핑을 유지한다. 원본 image 위치는 준비된 input_ids와 실제 모델 token ID/span/grid를 대조해 검증하며 token ID 숫자를 임의 고정하지 않는다.
- `vision_block_runtime.py`: full-length replay에 위 두 mask를 추가한다. clean capture, scoring, prefix 검증과 수치 정책을 재사용한다.
- `vision_block_metrics.py`: 명시된 profile의 조건 집합으로 정확한 row coverage를 검증하되 다섯 지표 공식은 변경하지 않는다. 기존 기본 조건 집합을 보존한다.
- `vision_block_analysis.py`: profile을 실제 실행, smoke gate, identity, 원시 행 검증 및 보고에 일관되게 전달한다. 다른 profile/model/manifest의 smoke를 재사용하지 못하게 한다. 보고서에는 실제 모델과 profile을 표시하며 beta 모델을 CE-only로 표시하지 않는다.

새 결과에는 profile, 조건 순서, 원본 image 위치 목록 또는 hash와 개수, site별 차단 Body 범위, 모델·코드·입력 identity를 기록한다. 원본 image payload가 없거나 잘못 배치되었거나 조건 집합이 불일치하면 실패한다. 새로운 orchestration/retry 시스템과 shared training/decoder 수정은 범위 밖이다.

수치 정책은 기존 BF16 base와 저장된 FP32 adapter, SDPA MATH only, matmul/cudnn TF32 모두 off, vision FlashAttention2, FP32 log-softmax를 유지한다. 로더 호출 뒤 실제 policy를 다시 적용하고 관측한다. 학습 설정을 새 기본값으로 바꾸지 않는다.

## 5. 지표와 출력

분모는 공통 site 수 S=327, target 수 N=12,843이다. 정답 y는 TEST annotation에서 독립적으로 얻으며 clean 예측을 정답으로 쓰지 않는다.

| 지표 | 공식 |
|---|---|
| 첫 토큰 변경 | 조건 argmax와 clean argmax가 첫 위치에서 다른 site 수 / S |
| 구간 내 변경 | 한 개 이상 argmax가 다른 site 수 / S |
| 전체 토큰 변경 | argmax가 다른 target 위치 수 / N |
| 평균 NLL | 모든 target의 −ln p(y) 합 / N, 자연로그·토큰 가중 평균 |
| ΔNLL | 조건 평균 NLL − 같은 실행의 완전한 clean 평균 NLL |

주 결과는 clean을 포함한 3행 × 5지표다. 두 mask 조건 모두 같은 완전한 clean과 비교한다. image_mask를 2번 조건의 기준으로 바꾸지 않는다. ΔNLL 음수도 보존한다. site별·token별 결과를 저장하고 중복/누락 없이 **38,529개 행(3×12,843)**을 검증한다. 이전 beta0.5 clean 결과는 진단 대조 자료이며 새 실행의 clean을 대체하지 않는다.

결과는 `results/<새 run-id>/`, 로그는 `logs/<새 run-id>/`, manifest는 `data/ce-vision-block-analysis/<새 run-id>/`에 둔다. 기존 결과와 심볼릭 링크를 보존하고 모델 파일은 기존 checkpoints 경로에서 읽는다.

## 6. 검증과 실패 처리

구현은 sub-agent에 위임하고 TDD, 독립 spec/quality 리뷰 및 해당 final review를 수행한다. root는 실행과 증거 비교를 담당한다.

집중된 테스트는 다음 주장에 직접 대응해야 한다.

- literal visibility 검사: 이미지 key는 END 전 query에서 계속 보이고 END 이후에서만 차단된다. 2번은 현재 Body만 추가 차단하며 다른 Body와 모든 START/END는 그대로다. 이미지 marker와 payload를 혼동하지 않는다.
- 입력·순서 검사: 모든 embedding/길이/3축 position/정답 위치가 동일하고 site 순서를 바꿔도 결과가 같다. 실제 decoder 진입점에서 mask 적용을 관측한다.
- live causal control: 테스트가 실제 attention의 K/V에서 **q ≥ e로 향하는 해당 image/Body 기여만** 변조해, 차단 조건의 불변성과 허용 조건의 반응을 검증한다. image embedding을 rollout 이전에 바꾸면 간접 prefix 경로도 달라지므로 이를 차단 실패 검사로 사용하지 않는다. full-length 계산에서 사용되지 않는 저장 K/V만 변조하는 테스트도 증거로 인정하지 않는다.
- 독립 tiny FP32 reference와 production의 대응 target logits를 `atol=1e-5, rtol=1e-5`로 비교한다. reference는 production mask builder를 재사용하지 않는다. 공유 decoder 수학 전체의 독립 검증이라고 부르지 않는다.
- profile별 정확한 coverage, 손계산 지표, 음수 ΔNLL, 기존 네 조건의 호환성, 잘못된 smoke/profile 차단을 검사한다.

실제 GPU 최소 검증은 기존 source3(1 site, 91 targets), source101(6 sites, 224 targets)로 시행한다. 세 조건의 21회 prefix 검사(36 layers×K/V=1,512 comparisons)와 반복 clean 7회(504 comparisons)를 요구한다. ordinary clean/capture/replay 및 반복 clean의 argmax 차이는 0, NLL 절대차 최대 ≤0.01 nat·평균 ≤0.001 nat다. 수치 기준을 결과를 보고 완화하지 않는다. 새로운 두 mask의 runtime mask witness도 필요하다.

clean 비교 쌍은 ordinary↔captured, ordinary↔replay, captured↔replay다. smoke에서는 두 번의 ordinary, 두 번의 captured, 두 번의 replay도 각각 비교한다. 각 비교의 최대·평균 NLL 절대차는 해당 sample의 선택된 모든 target 위치에서 계산하며 sample 평균을 다시 평균하지 않는다. 본실험은 각 sample의 앞 세 비교와 site별 prefix/visibility gate를 검사한다.

smoke 통과 후에만 고유 run-id로 GPU 1개에서 100개를 detached 실행한다. 모든 site/조건에서 동일한 gate를 적용한다(327×3×36×2=70,632 prefix K/V comparisons). 모든 행을 독립 검산하고 즉시 결과를 보고한다. 필수 branch 완료 검증이 적용될 때 프로젝트 전체 pytest를 한 번 실행하며 동일 revision에서 불필요한 반복 broad test/review를 하지 않는다. 실패 시 로그를 보존하고 최소 수정·causal regression·새 run-id로 재시도한다.

## 7. 선택한 접근과 남은 불확실성

선택한 local END-onward mask는 기존 clean-prefix invariant와 사용자 정의를 유지한다. 처음부터 모든 query의 image 접근을 막으면 이전 hidden state가 바뀌며, image-off latent 재생성은 사용자 선택과 다르다. 두 방식은 이번 실험으로 채택하지 않는다.

현재 증거: 기존 beta0.5 네 조건 실행은 같은 100개/S327/N12843에서 clean argmax/NLL 차이 0과 정확한 prefix 보존을 관측했다(`results/b05-block-main100-v2-20260922-044315/main100/controller-validation.json`). 실제 full-length decoder가 additive mask를 SDPA에 전달하는 call path는 해당 worktree에서 확인했다. 이는 새 이미지 mask의 실행 검증을 대신하지 않는다.

아직 확인되지 않은 것은 새로운 image mask의 구현 정확성과 실제 BF16에서의 통과 여부다. 설계 단계에서는 product 변경이나 새 GPU probe를 하지 않았다. 이 불확실성을 남긴 채 구현을 준비하는 비용만 승인 대상으로 제시하며, 새 causal test와 GPU smoke 통과 전에는 본실험 결과를 만들거나 성공을 주장하지 않는다. runtime gate를 면제하는 위험 수용은 요청하지 않는다.

여기서 받아들일 불확실성은 새 mask의 실증 검증 전에 설계·구현 비용을 쓰게 된다는 점이다. 새 mask의 first-nail 실험은 구현 후 본실험 전의 필수 gate로 둔다. 아직 없는 새 mask 경로를 만들기 위한 별도 prototype/GPU 실험을 이번 문서 작성 중 실행하지 않았으며, 서면 승인 요청에서 이 미검증 상태를 명시한다.

## 8. 설계 확정 검토와 정확한 commitment v1

동결 proposal v1은 `results/image-block-analysis-design-20260922/design-proposal-v1.md`, SHA256 `d8351b131098d17ffeafc6de6b7bea6fbf2719012de1e11b96e3c1ce0a8a8b85`다. 다음 검토는 동일한 동결본을 대상으로 수행했다. 검토 후 섹션 1–7의 문제·범위·인터페이스·동작 변경은 없다.

**hate**

- root: 새 mask가 첫 토큰 predictor인 END의 이미지 접근을 남겨 두거나 clean prefix/visual 생성까지 바꾸면, 결과는 요청한 직접 접근 차단 효과가 아니다.
- first_nail: tiny 실제 decoder와 지정한 실제 single/multi 샘플에서 END 및 이후 평가 predictor의 layer별 effective mask, 정확한 차단 key, embedding/position 동일성, pre-END K/V 동일성을 관측한다. prefix를 건드리지 않는 query 한정 live K/V 변조로 blocked-edge 불변성과 visible-edge positive control을 확인한다.
- 상태: **새 mask 실험은 미실행**. 해당 경로의 구현 전이며 별도 prototype/GPU premise probe를 승인받지 않았다. 기존 full-length 실행 증거와 실제 call path는 근거 자료이지만 새 mask의 실증 증거는 아니다. 새 causal test와 실제 smoke를 본실험 전 필수 gate로 둔다.

**macrothink 적용 여부:** 사용자 답변으로 모델·개입 시점·clean 생성 유지·직전 Body만 차단하는 범위가 해결됐다. 현재 남은 문제는 대안적 목적 선택이 아니라 구현·실행 증거이므로 추가 fresh direction panel은 적용하지 않았다.

**prism:** root controller가 두 구별되는 렌즈로 검토했다. 독립 코드 리뷰로 표시하지 않는다.

- 과학적 개입/정보 경로: 설계 수준 pass. 직접 edge만 차단하고 간접 image 경로가 남는다는 해석 범위를 명시했다.
- 수치/실행 계약: 실제 검증 전 unclear. 기존 BF16 결과는 새 조건의 mask 적용과 profile gate를 증명하지 않는다.
- 남은 한 질문: 독립 리뷰를 마친 새 profile이 지정한 tiny/실제 샘플에서 prefix·target을 보존하고 실제 edge를 정확히 차단하는가? 구현자 테스트, 독립 reviewer, controller GPU smoke가 각각 필요한 증거를 제공한다. 두 판정을 평균하거나 미검증 부분을 통과로 바꾸지 않는다.

**mandela:** Wrong null hypothesis 위험은 이 결과를 전체 시각 정보 제거로 해석하는 것이다. clean history/다른 block/marker/teacher-forced 텍스트에 간접 정보가 남는다. 현 주장을 국소 직접 접근 민감도로 제한하고, NLL 정답을 annotation에서 가져오는 것으로 독립성 범위를 유지한다. semantic accuracy나 학습 데이터 오염의 부재를 검증했다고 주장하지 않는다.

**shower:** `/root/image_local_spec_coldread`가 주변 문맥 없이 전체 초안을 읽어 minor gaps를 보고했다. 승인 순서, target 목록의 규범, 기존 CLI/조건 계약, 수치 비교 쌍·집계 단위를 보완했다. 이 보완은 proposal 동결 전에 이루어졌으며 실험 범위를 바꾸지 않았다. 보고·증거는 `results/image-block-analysis-design-20260922/`에 보존한다.

**Superpowers commitment v1:** 섹션 1–7을 그대로 채택한다. beta0.5 checkpoint212, 같은100/S327/N12843, clean 및 두 국소 mask, 원본 이미지 기반 clean visual 생성·pre-END 상태 보존, START/END 입력 유지, full-length/동일 수치 정책, 같은-run clean 기준 다섯 지표, profile 호환성과 엄격한 coverage, 위임 구현·독립 리뷰·새 causal/smoke gate가 승인 대상의 정확한 범위다. 새 mask 실증 전 구현 준비 비용을 쓰는 불확실성을 명시적으로 승인 요청에 포함한다. 이미 부여된 risk waiver는 없으며 runtime 검증 면제도 요청하지 않는다. 서면 승인 후 다음 단계는 구현 계획 작성이고, 기존 sub-agent 실행 선택은 유지한다.
