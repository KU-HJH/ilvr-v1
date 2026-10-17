# DIL-BCE 계열의 검증된 ILVR 학습 경로 이식

상태: 대화 설계 승인 완료, 작성된 명세의 사용자 검토 대기.
날짜: 2026-09-28.

## 1. 목적과 승인 범위

`/raid/workspace/temp2/ILVR/.worktrees/ours1-dil-bce`의 DIL-BCE 목적함수를
현재 ILVR의 검증된 context-preference 실행 경로로 이식한다. 사용자는
별도 worktree 사용, 아래 8개 variant 전부와 `voff` entropy 선택 포함,
원본 수식·gradient·reduction 보존, 현재 학습·checkpoint·평가 경로 연결을
승인했다. 실행·검증 설계 제시 후 `Ok`로 명세 작성을 승인했다.

작업 경로는 `/root/workspace/ILVR/.worktrees/dil-bce-port`, branch는
`feat/dil-bce-port`다. 기반은 `voff-reproduction-20260917`의
`51e11733084275bb2b1b2079924368d3cd636d9b`이며, 정책에 지정된
`1a63667bc9db3f54c5cfc350e874797e790e952a`의 descendant다.
이식은 기존 clean/V-off 실행 위에 새 목적함수를 추가하는 방식이다.

성공 기준은 설정 선택부터 loss/backward, 저장·동일 설정 resume, PEFT
평가 준비·생성까지 연결하는 것이다. CPU 검증, coordinator dry-run,
실제 두 노드 GPU 검증은 각각의 증거 수준으로 보고한다.

다음은 이번 이식 범위에 포함하지 않는다: 원본 trainer 전체 교체,
원본 과거 checkpoint의 training resume 변환, 기존 결과·weights 복사,
8개 전체 재학습, TEST400 성능 재현, latent causal 분석 도구 이식,
새로운 학습 orchestration 또는 환경 의존성 재설치.

명세 승인과 계획·구현 방법 선택은 아직 이루어지지 않았다. GPU 실행은
구현 후 구체적인 명령·규모·입출력을 제시하고 별도로 승인받는다.

## 2. 원본과 기반의 확인된 차이

원본 loss, entropy helper, trainer, 설정·metadata 및 관련 테스트는
미커밋 변경을 포함한다. Git commit만으로 원본을 특정하지 않는다.
20개 관련 파일의 SHA-256을
`results/dil-bce-port-workflow/source-identity.json`에 기록했다.
실제 구현 시 이 identity를 확인하고 원본 변경이 있으면 차이를 먼저 검토한다.

기반 trainer의 `_compute_context_preference_loss`는 clean selected logits와
sidecar별 평균 GT log-probability/entropy를 받는다. 기본 BCE는 이 점수만으로
계산할 수 있지만 top-k/mass는 OFF 어휘 분포 정보가 더 필요하다.

기반 `stage2_shared_prefix.py`는 sidecar decoder와 LM-head 계산을 기존
non-reentrant checkpoint 안에서 수행한다. 이 경계를 유지하면서 token별
목적함수 통계를 반환하도록 확장한다. 기반의 독립 context-group embedding
lookup, live clean KV, clean projection/CE-before-sidecar 순서를 보존한다.

기존 coordinator는 `src/multinode.py`에 있으며 worktree 상대 경로를 지원한다.
새 환경 변수는 `src/multinode_runtime.py` allowlist와 worker launcher까지
연결해야 한다. 기존 coordinator와 평가 준비 도구를 재사용한다.

## 3. 지원 설정과 variant

loss selector는 `dpo | dil_bce_rf | dil_bce_rf_topk`이고 기본은 `dpo`다.
새 loss를 선택하지 않으면 기존 동작을 유지한다.

| 실행 예시 이름 | selector | k | mass weight | entropy selection |
|---|---|---:|---:|---|
| bce | dil_bce_rf | 없음 | 0 | 없음 |
| topk-k8 | dil_bce_rf_topk | 8 | 0 | 없음 |
| topk-k5-mass | dil_bce_rf_topk | 5 | 1 | 없음 |
| topk-k8-mass | dil_bce_rf_topk | 8 | 1 | 없음 |
| topk-k50-mass | dil_bce_rf_topk | 50 | 1 | 없음 |
| topk-pct1-mass | dil_bce_rf_topk | floor(V/100) | 1 | 없음 |
| topk-pct10-mass | dil_bce_rf_topk | floor(V/10) | 1 | 없음 |
| topk-pct1-mass-entropy20 | dil_bce_rf_topk | floor(V/100) | 1 | clean 상위 20% |

원본 V=151668에서는 비율 예시가 각각 k=1516, 15166이다. 예시는 실제
resized vocabulary를 확인해 정수 k로 resolve하고 두 노드 실행 기록에 남긴다.
이는 vocabulary 개수의 비율이며 누적 probability-mass cutoff가 아니다.

추가 CLI는 다음으로 고정한다. 기존 beta/weight/target-scope CLI는 재사용한다.

- `--step_preference_loss_type`
- `--step_preference_topk`
- `--step_preference_lambda_topk_mass`
- `--step_preference_entropy_top_percent`
- `--step_preference_entropy_source` (`clean | voff`, 기본 `clean`)

대응 환경 변수는 `STEP_PREFERENCE_LOSS_TYPE`, `STEP_PREFERENCE_TOPK`,
`STEP_PREFERENCE_LAMBDA_TOPK_MASS`, `STEP_PREFERENCE_ENTROPY_TOP_PERCENT`,
`STEP_PREFERENCE_ENTROPY_SOURCE`다. 실행 예시는 실제 coordinator dry-run에
연결되는 직접적인 명령으로 제공하고 별도 campaign framework를 만들지 않는다.

지원 조건은 enabled Stage 2B `context_preference`, visual-only,
`clean_prefix`, independent branches, shared prefix ON, latent-body attention
차단, START/END 유지·supervision, original image 접근 유지, CE=1,
entropy-loss weight=0, preference margin=0, reference-free다.
`immediate_text`와 `until_next_latent`를 모두 지원한다.
기반 DPO의 다른 기존 조합은 기존 검증 계약을 유지한다.

새 selector에서 지원하지 않는 조합은 forward 전에 실패해야 한다. k는
실제 어휘 크기에 대해 `2 <= k < V`인 정수이고 mass weight는 finite >=0이다.
DPO/기본 BCE에서 top-k·mass·entropy-selection 설정을 무시하지 않고 거부한다.
beta는 finite >0, preference weight는 finite >=0이며 새 실험 기본값은
각각 0.5, 1.0이다. weight=0도 원본 수식 검증에 포함한다.

## 4. 목적함수와 gradient 계약

step j의 대상 GT token 평균 log-probability를 clean에서 sC_j,
OFF에서 sO_j라 한다. 기본 BCE는 다음과 같다.

`L = CE + lambdaP * mean_j[softplus(-beta*sC_j) + softplus(beta*sO_j)]`

평균 token score에 softplus를 적용한다. token별 softplus 평균으로 바꾸지 않는다.
clean과 OFF score 모두 gradient를 유지한다. 별도 negative 문장, frozen
reference policy, density ratio를 추가하지 않는다.

Top-k 모드에서는 OFF logits의 detached top-k indices로 token별 집합 S를
고르고, 선택된 live OFF logits의 conditional softmax를 q로 정의한다.

`negative_t = sum_{v in S} q_v * softplus(beta * log(q_v))`

q 가중치와 log(q) 양쪽 모두 gradient를 유지한다. positive는 기본 BCE의
clean 항이다. 기본 BCE의 GT-negative 또는 DPO 항을 추가로 더하지 않는다.

같은 OFF-selected S에 대해 full-vocabulary 확률 질량 mC, mO를 구한다.
mass 항은 `KL(Bernoulli(stopgrad(mC)) || Bernoulli(mO))`다. clean target은
현재 clean forward에서 얻고 gradient를 차단한다. OFF selected/complement
집합 모두 gradient를 유지한다. support identity는 고정하지 않는다.

`L = CE + lambdaP * mean_j[positive_j + mean_t(negative_t)]`
`    + lambdaM * mean_j[mean_t(mass_t)]`

mass coefficient는 lambdaP와 독립이다. lambdaP=0, lambdaM>0이면 mass는
활성이다. token 평균 후 step 평균 순서를 유지한다. FP32 연산 및 원본의
안정적인 group-mass 계산을 보존한다. finite/빈 target/BF16 입력을 검증한다.

분산에서는 기반의 sample-local step 평균, sampler, partial accumulation,
zero-loss padding과 configured-global-batch 분모를 유지한다. 원본의
latent-balanced sampler에 의존한 '모든 rank의 step 수가 같다'는 가정을
reference_random으로 가져오지 않는다. rank별 step 수가 다른 CPU oracle로
sample 가중과 마지막 partial update의 분모를 별도로 확인한다.

## 5. Entropy filtering과 sidecar 데이터 흐름

필터는 `dil_bce_rf_topk`에서만 허용한다. 비율 P는 finite `0<P<=100`이며
생략 시 OFF다. `voff` source는 비율을 명시해야 한다.

각 sample의 모든 유효 step target predictor 위치를 모아 선택된 source의
full-vocabulary Shannon entropy로 정렬한다. `ceil(T*P/100)`개를 고르고
동률은 원래 token 위치 순서로 해결한다. 소수 비율 quota는 원본의 정확한
계산을 보존한다. ranking은 detached다. P=100은 OFF와 같은 경로를 사용하며
추가 entropy-selection 계산을 하지 않는다.

동일 mask를 positive GT score, top-k negative, mass에 적용한다. positive는
선택된 clean GT log-probability의 평균에 softplus를 적용한다. 선택된 위치가
없는 step은 0을 기여하지만 원래 step 분모에는 남긴다. CE는 모든 기존
supervised 위치에서 유지한다. 선택되지 않은 위치도 공유 문맥을 통해
간접 gradient를 받을 수 있다.

데이터 흐름은 다음과 같다.

1. 기존 clean forward와 CE/selected-logit projection을 완료한다.
2. top-k 모드에서만 기존 sidecar checkpoint에 step별 정렬된 detached clean
   분포 입력과 loss 설정을 전달한다. 기존 clean selected logits를 재사용한다.
3. checkpoint 안에서 OFF projection을 한 번 수행하고 token별 top-k negative,
   mass, GT logp 및 필요한 진단 통계를 계산한다. `voff` 선택에 필요한 full
   entropy도 여기서 계산한다. decoder 또는 LM-head 호출을 추가하지 않는다.
4. checkpoint 출력은 token별 scalar 통계로 제한한다. OFF `[T,V]` logits를
   새로운 trainer 반환값으로 유지하지 않는다. clean mass 입력 detach는 clean
   positive와 clean CE의 live 경로를 차단하지 않는다.
5. trainer가 sample별 선택 mask를 생성하고 원래 token/step reduction을 적용한다.

branch/step/target ID·predictor 위치의 일치 여부를 검증한다. checkpoint 재실행은
같은 입력과 support 선택 의미를 사용하고 live clean KV와 latent gradient를
유지한다. 기존 DPO 경로에는 새 어휘 통계와 새 결과 schema를 요구하지 않는다.

이는 OFF logits의 checkpoint 외부 보관을 줄이는 설계이지 전체 학습 메모리의
상수 크기 보장이나 원본과 같은 peak memory라는 주장은 아니다. 기존 clean
selected logits 및 autograd 상태 비용은 남는다.

## 6. 저장·resume·평가 계약

기반의 model config 저장 경로에 versioned `ilvr_context_preference` 객체를
추가한다. 새 모드의 final artifact와 numbered checkpoint에 같은 identity를
저장한다. namespace schema version은 1이며 원본 schema 13/14를 복사하지 않는다.

identity에는 selector, formula version, beta, preference/CE/entropy/mass weights,
margin, resolved k, entropy percentage/source, target scope, prefix/branch,
mask·boundary·image 의미와 sidecar 실행 설정을 기록한다. formula version은
`dil_bce_rf_v1` 또는 `dil_bce_rf_topk_v1`이고 entropy 선택/reduction 의미도
versioned identity에 포함한다. k가 없는 mode는 null로 기록한다.

동일 run resume에서 이 identity가 달라지면 optimizer state를 진행하기 전에
거부한다. 새 BCE 모드인데 identity가 없거나 알 수 없는 version/field가 있으면
실패한다. 기존 identity 없는 checkpoint는 기존 DPO 로딩 규칙을 유지하되
자동으로 BCE로 해석하지 않는다. 새 학습의 Stage1 초기화에는 BCE identity가
필요 없고 기존 Stage1 provenance 검증을 유지한다.

기존 PEFT schema, sparse token rows, original base identity, ZeRO shards,
sampler/runtime/RNG resume 검증을 약화하지 않는다. config.json은 기존
checkpoint collection·inference allowlist에 있으므로 기존 전달 경로를 활용한다.
metadata round-trip과 inference preparation이 새 identity를 보존하는지 검사한다.

새 artifact는 existing PEFT inference preparation으로 original base와 결합하고,
기존 latent generation·평가 entrypoint를 사용한다. 평가 시 DIL loss를 계산하지
않는다. 과거 source checkpoint 포맷의 자동 migration은 지원 범위가 아니다.

## 7. 기본 실행 프로파일

canonical `AGENTS.md`와 `documents/training-defaults.md`가 우선한다. 기반
worktree AGENTS에 남아 있는 historical global-batch-8 문구를 새 기본값으로
사용하지 않는다. 이식 worktree의 지침은 canonical 정책 참조와 모순되지 않게
정리한다.

- 2 nodes × 2 GPUs = 4 ranks, microbatch 1, accumulation 4, global batch 16.
- LR 5e-5, seed42, warmup10, BF16, 기존 AdamW/linear schedule.
- beta0.5/preference1/CE1/entropy0, 기본 target `immediate_text`.
- `reference_random`, `concatenated-v1`, 승인된 crossing-token exclusion.
- DoRA r32/alpha64/dropout0.05, 기존 decoder target modules, latent8.
- decoder SDPA, vision FA2, sidecar Flex/query budget1024, native checkpoint OFF,
  latent와 기존 sidecar checkpoint ON, ZeRO-2, max train sequence8192, workers0.
- original Qwen2.5-VL-3B + reproduced Stage1
  `stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431` 초기화.
- direct-from-Stage1 경로와 기존 phase 설정 의미를 유지한다. 3379 retained
  samples의 기본 전체 exposure는 212 updates이며 실제 데이터·sampler로 확인한다.

기존 환경을 source한 뒤 전체 applicable profile을 명시적으로 적용한다.
원본 역사적 2-rank/accumulation8/old frozen Stage1 설정을 예시에 복사하지 않는다.
새 모드를 선택하지 않은 기존 실행의 fallback을 광범위하게 변경하지 않는다.

TEST400 평가 기본값은 greedy, temperature0/top_p1, max new tokens4096,
max sequence128000, repetition penalty1, no vision ablation,
`external_full_latent_block`다. smoke subset은 성능 평가와 구분한다.

새 worktree의 `venv`, `data`, `checkpoints`는 기존 canonical 자원을 가리키는
symlink를 사용한다. 모델 artifact는 `checkpoints/`, 평가 staging은
`checkpoints/evaluation-staging/`, 결과는 worktree `results/`, 로그는 worktree
`logs/`에 둔다. 원격은 같은 reviewed revision을 같은 worktree 상대 경로에
준비하고 기존 coordinator의 양 노드 preflight를 통과해야 한다.

## 8. 검증과 완료 판정

### CPU 구현 증거

1. 원본 pure loss/helper와 동일 입력·seed에서 값과 입력 gradient 비교. 원본
   source identity를 함께 기록한다. 지속적인 repo 테스트는 `/raid` 경로에
   의존하지 않는 독립 수식 oracle과 fixture를 사용한다.
2. BCE positive/negative 분리, q의 양쪽 gradient, clean mass detach,
   complement gradient, equal-step reduction, mass-only, 빈 target,
   extreme finite 값, BF16 casting, k 범위와 nonfinite 설정 검사.
3. Entropy sample 격리·stable tie·quota·unselected-step 분모, clean/voff source,
   OFF/100% bit-exact parity 및 부분 선택의 loss/gradient oracle.
4. 실제 tiny Qwen forward/backward로 입력 정렬, live KV/latent gradient,
   clean-before-sidecar 순서, 독립 embedding lookup, checkpoint on/off의
   수치·gradient 일치, 기존 대비 decoder/LM-head 추가 호출 0을 검증한다.
   모델 전체의 원본과 대상 간 동등성은 별도 증거 없이 주장하지 않는다.
5. DPO 비선택 경로 회귀, 서로 다른 step 수의 sample 분산 가중,
   zero-loss padding 및 partial accumulation 분모 검증.
6. 설정 parser, 8개 예시와 voff 선택의 coordinator dry-run에서 양 rank-node
   env/CLI에 동일한 resolved 값이 전달되는지 확인한다.
7. 저장 identity round-trip, changed-setting resume 거부, historical DPO,
   Stage1 초기화, PEFT 준비·평가 로더 계약을 검사한다.

원본/대상 loss parity는 동일 logits 조건에서 검증한다. training sampler,
tokenization, 초기 weights와 attention 경로가 다르므로 원본 55% 재현이나
학습 trajectory 동등성을 이 검증으로 주장하지 않는다.

### 실제 GPU 증거

CPU 검증·독립 코드 검토 후 실제 명령, run IDs, sample IDs, update 수,
예상 자원과 출력 경로를 제시해 별도 실행 승인을 받는다. 승인 전 GPU 학습,
checkpoint 생성, inference model 준비·전송, 평가를 실행하지 않는다.

GPU acceptance에는 기존 coordinator를 통한 네 rank의 실제 참여, 유한
loss/backward/update, checkpoint 저장·같은 설정 resume, original base를
사용한 PEFT 평가 준비, 소수 sample 생성·scoring의 연결 증거가 필요하다.
BCE와 top-k 계열 및 entropy source 경로의 실제 검증 범위를 명확히 기록한다.
smoke로 실행하지 않은 조합은 CPU 검증만 된 것으로 보고한다.

preflight가 실패하거나 자원이 충돌하면 진행하지 않는다. 단일 노드 fallback,
임의 sequence 축소, 다른 loss로 대체하지 않는다. 실제 GPU 증거 전에는
'두 노드 학습/평가가 확인됐다'고 보고하지 않는다.

### 개발 완료와 handoff

TDD, 선택된 구현 방법, 독립 specification/quality review, 수정 검토를 따른다.
사용자가 구현 방법을 아직 선택하지 않았으므로 계획 단계에서 선택받는다.
claim-specific 검증과 branch finishing에 필요한 project CPU full suite를
마지막 reviewed revision에서 한 번 수행하고 겹치는 증거로 재사용한다.
명령은 venv와 vendored PYTHONPATH를 사용하며 project `tests`를 대상으로
`-m 'not gpu'`를 적용한다. 실제 실행 명령과 skip/failure 이유를 기록한다.
외부 upstream Transformers 전체 test suite로 범위를 확장하지 않는다.

코드 검증 완료, GPU smoke 완료, 전체 benchmark 완료를 각각 구분한다.
승인되지 않은 GPU 단계가 남으면 코드 이식 결과와 남은 검증을 명시하며
end-to-end 완료로 표시하지 않는다. merge/PR/keep 결정은 사용자가 한다.

## 9. 변경 경계와 현재 증거

예상 변경 지점: `src/step_preference/` loss·entropy·config helpers,
`src/stage2/trainer.py`, CLI resolver, 기존 model config identity와 resume 검사,
vendored Qwen shared-prefix scoring/옵션 검증, `run_stage2_training.sh`,
`src/multinode_runtime.py`, 해당 테스트와 8개 실행 예시 문서.
기존 inference 도구의 수정은 새 artifact 연결에 꼭 필요한 경우로 한정한다.

2026-09-28 worktree 생성 직후 기존 config/context preference/clean-first 테스트:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
./venv/ilvr-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/step_preference/test_config.py \
  tests/step_preference/test_context_preference.py \
  tests/step_preference/test_clean_first_order.py -m 'not gpu'
```

결과: 84 passed in 9.85s. 로그: `logs/dil-bce-port-workflow/baseline.log`.
이는 이식 전 제한된 baseline 증거이며 이식 구현 검증 또는 full-suite 결과가 아니다.

명세 self-review는 placeholder, 설정·수식·분모 모순, 범위, resume/실행 승인
경계를 확인한다. 작성된 명세를 사용자가 승인하면 writing-plans로 전환한다.
