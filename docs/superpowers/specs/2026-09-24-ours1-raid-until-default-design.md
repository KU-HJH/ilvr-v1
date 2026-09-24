# ours1 RAID until_next_latent 동기화 설계

상태: 동작 설계 승인 완료. 구현 방식 제한은 후속 사용자 지시에 따라 제거.
대상: `/root/workspace/ILVR/.worktrees/ours1`.
기준 HEAD: `5557bdd1ef41ea8e5354a2ac1990b636f0d5cf19` 및 현재 작업 트리의 기존 변경.

## 1. 목적과 승인 범위

로컬 ours1의 학습 텍스트, preference target token, activation checkpointing 위치를
RAID의 `ours1-untilnext-s1dora-beta05-coef1-1epoch-2gpu-20260917` 기록에 맞춘다.
새 context-preference 학습의 target scope 기본값은 `until_next_latent`로 한다.

사용자는 기본 경로 교체 제안에 “그렇게 하자. 그리고 until_next_latent를 default로 해놓자.”라고
승인했다. 이어 기본값 일관성, 명시적 immediate_text와 resume 존중, 토큰 비교 및
실제 forward/backward 검증을 포함한 설계의 문서화에 “Ok”라고 승인했다.
이 승인은 명세 작성까지이며, 작성된 명세와 이후 구현 계획의 승인과 구분한다.

## 2. 선택한 접근

기존 ours1 경로를 수정한다. 별도 선택 프로필을 추가하는 대안은 기존 동작을 선택하기
쉽지만 새 기본값과 별도 경로를 함께 관리해야 한다. 사용자는 기본 경로 교체를 선택했다.

기존 `until-temp2-alignment-20260923` 및 `preference-alignment-completion-20260923`의
관련 구현과 테스트를 선별 재사용한다. 이들 작업은 sampler, DoRA, attention 등의
추가 변경을 포함하므로 전체 이식은 범위를 벗어난다. 후속 sidecar checkpoint ON
최적화도 이번 요구의 OFF 설정과 다르므로 그대로 재사용할 수 없다.

## 3. 기준 증거와 주장 한계

RAID root는 `/raid/workspace/temp2/ILVR`이다.

- `results/step-preference/ours1-untilnext-s1dora-beta05-coef1-1epoch-2gpu-20260917/`
  아래의 `run-config.json`, `off-train-command.txt`, `off-status.json`을 실행 설정의 근거로 삼는다.
- `.worktrees/ours1/src/stage2/data.py`의 `_assistant_text`는 조각을 그대로 연결한다.
- `.worktrees/ours1/src/step_preference/spans.py`의 `derive_reasoning_step_spans`는
  latent 제어 토큰을 기준으로 target 구간을 계산한다.
- `.worktrees/ours1/documents/ours1-selective-latent-checkpointing-20260907.md` 및
  run의 launch/log 기록은 native OFF, cache-free latent checkpoint ON,
  final clean과 sidecar checkpoint OFF의 근거다.

run 기록은 `uncommitted_source: true`다. 현재 RAID 소스가 당시 소스 전체와 동일하다는
증거는 없다. 현재 보존 소스와의 비교, 당시 기록과의 설정 일치, 로컬 실행 검증을
각각 보고한다. 당시 학습 전체의 수치적 동일성이나 정확도 재현은 완료 조건이 아니다.

## 4. 텍스트와 target 데이터 흐름

새 until 경로에서 sequence plan의 text content와 latent 제어 블록을 순서대로
연결한다. text/text 사이에 공백이나 줄바꿈을 추가하지 않는다. 데이터에 원래 포함된
공백과 줄바꿈은 보존한다. 같은 processor와 chat template로 전체 입력을 토큰화한다.

각 latent의 target은 반열린 token 구간 `[END + 1, next START)`다.
마지막 latent는 `[END + 1, assistant content end)`이며 최종 답변을 포함한다.
assistant 뒤의 chat 종료 제어 토큰을 content에 임의로 포함하지 않는다.
logit predictor 위치는 target 위치에서 1을 뺀 구간이다.

until 구간 내부의 text/text 경계를 걸치는 토큰은 target에 포함한다. 내부 plan 경계만을
이유로 샘플을 거부하거나 토큰을 제외하지 않는다. latent 제어 블록, assistant 경계,
구간 순서와 빈 target에 대한 검증은 유지한다. CE label은 변경된 입력 토큰에 맞추되
기존 CE supervision 규칙을 유지한다.

담당 코드: `src/step_preference/spans.py`, `src/stage2/data.py`, `src/main.py`.
명시적 `immediate_text`는 기존 선택지로 유지하며 그 target 정의를 until로 바꾸지 않는다.

## 5. Activation checkpointing과 KV 연결

| 실행 구간 | 새 until 기본 경로 |
| --- | --- |
| native decoder layer checkpointing | OFF |
| cache-free latent 반복 | ON |
| final clean forward | OFF |
| V-off sidecar forward | OFF |

latent prefix의 embeddings, attention mask, positions와 필요한 tensor 입력을
checkpoint 함수에 명시적으로 전달한다. non-reentrant checkpoint와 RNG 보존으로
DoRA dropout 및 backward 재계산의 입력을 유지한다. 반복문 closure가 다음 반복의
상태를 참조하지 않도록 기존 검증된 구현을 사용한다.

final clean에서 생성한 KV는 V-off의 prefix로 전달되고 gradient가 clean 경로까지
연결되어야 한다. detach하거나 no_grad로 바꾸지 않는다. 체크포인팅 구간 변경을 위해
KV 표현을 무조건 교체하지 않는다. 로컬 명시적 KV tensor 경로를 유지할 경우에는
동일한 공유·gradient 계약을 실제 실행으로 검증한다. 이 검증은 RAID의 DynamicCache
객체 구현까지 동일하다는 주장을 의미하지 않는다.

담당 코드: `src/main.py`, Qwen의 `modeling_qwen2_5_vl.py`,
`stage2_latent_rollout.py`, `stage2_shared_prefix.py` 및 대응하는 modular 소스가 있는 경우 그 소스.
새 until 경로에서 native checkpoint와 selective latent checkpoint를 함께 요청하면
명확한 설정 오류로 처리한다. memory 최적화를 이유로 sidecar checkpoint를 자동으로 켜지 않는다.

## 6. 기본값, 명시 설정 및 resume

launcher, CLI, config 및 실제 collator에 전달되는 새 context-preference의 생략된
scope를 일관되게 `until_next_latent`로 해석한다. 기본값을 바꿨다는 이유로 CE-only,
preference-off, Stage 1, Stage 2A 및 다른 objective의 기존 진입점이 거부되어서는 안 된다.
적용되지 않는 경로의 scope는 실행 모드에 맞게 정규화하고, 실제로 지원하지 않는
명시적 조합에 대한 오류 검증을 유지한다.

명시적 `immediate_text`와 기존 resume의 저장 설정은 우선한다. resume에 필요한
과거 rendering/checkpoint 설정이 충분히 기록되지 않았다면 새 기본값을 조용히
적용하지 않고 부족한 정보를 알린다. 과거 run 결과 및 checkpoint metadata는 수정하지 않는다.

담당 코드: `run_stage2_training.sh`, `src/utils_deepseed.py`,
`src/step_preference/config.py`, `src/main.py`와 필요한 multi-node 환경 전달 경로.
ours1의 사용 문서에는 이 기본값을 명시한다. canonical의 다른 worktree까지 기본값을
일괄 변경하는 작업은 포함하지 않는다. 이번 사용자 지시가 ours1의 이전 immediate_text
기본값보다 우선한다.

## 7. 범위와 작업 보존

현재 ours1에는 기존 미커밋 코드와 문서가 있다. 구현 단계에서 이를 보존한 작업 기준을
만들고 변경 전후 diff를 분리한다. 기존 변경을 임의로 되돌리거나 함께 커밋하지 않는다.

이번 작업은 위 세 동작과 관련 기본값·문서·테스트를 수정한다. V-off의 START/END
차단 정책, loss 정의, sampler, optimizer, 모델 초기화까지 RAID와 동일화한다는
범위로 확장하지 않는다. 따라서 완료 후에도 전체 preference 알고리즘이 RAID와
완전히 같다고 보고하지 않는다.

새 preference 실험의 beta 0.5와 weight 1.0, 두 노드 네 rank 등의 운영 정책은 계속
적용된다. 이번 설계는 실제 학습이나 평가 실행을 요청하지 않는다.

## 8. 검증과 완료 조건

1. **토큰 비교:** 같은 processor와 고정 입력으로 로컬 결과와 보존된 RAID 소스의
   assistant text, input IDs, target IDs 및 predictor indices를 비교한다.
   인접 text 조각, 경계 결합 토큰, 원래 공백·개행, 다중 latent 및 최종 답변을 포함한다.
   서로 다른 tokenizer 입력을 사용한 비교는 동등성 증거로 사용하지 않는다.
2. **실행 증거:** 작은 모델의 forward/backward에 호출 계측을 넣어 latent만 checkpoint
   대상이고 final clean과 V-off는 대상이 아닌지 확인한다. native 설정은 실제 Trainer와
   모델 enable 경로까지 추적한다. 단순 config 값이나 테스트 이름만으로 판정하지 않는다.
3. **수치·gradient:** 동일 입력과 RNG 상태에서 selective checkpoint OFF/ON의
   logits, loss와 trainable gradients를 비교하고 허용 오차를 보고한다. V-off loss에서
   clean prefix KV까지 gradient가 전달되는 것도 별도로 확인한다.
4. **설정 회귀:** 새 기본값, 명시적 immediate_text, CE-only/preference-off,
   다른 stage, resume 저장 설정 및 충돌 설정 오류를 검사한다. launcher dry-run과
   config/collator 호출을 함께 확인한다.
5. **리뷰와 최종 확인:** implementer와 별개의 리뷰어가 명세 준수와 품질을 검토한다.
   수정은 사용자가 선택한 구현 방식에 따라 처리한다. 검토된 revision에서 fresh verification을 실행하고
   실제 명령, 결과, diff 및 commit/hash를 기록한다.

Python과 테스트는 repository의 `venv/ilvr-venv`를 사용한다. CPU 검증 결과를 production
GPU 또는 full-training 검증으로 확대하지 않는다. 실제 training smoke가 추가로 필요해지면
기존 두 노드 네 rank coordinator 정책을 따른다. 발견된 blocking 차이는 해결하거나
미검증으로 보고하며 테스트가 없는 항목을 통과로 기록하지 않는다.

## 9. 다음 단계

사용자가 명세 검토 후 `superpowers:writing-plans`를 요청하여 구현 계획을 작성했다.
계획 검토 시 Subagent-driven 또는 Native 중 구현 방식을 선택한다. 선택된 방식에
따라 구현하고 독립 리뷰와 최종 검증을 수행한다. 아직 구현, 학습 또는 평가를 시작하지 않았다.
