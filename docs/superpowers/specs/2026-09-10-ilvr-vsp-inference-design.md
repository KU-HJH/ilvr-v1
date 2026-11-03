# design commitment v1 — VSP 데이터 준비와 ILVR 추론 연결

작성일: 2026-09-10. 상태: 최종 설계 v1 및 명시한 한계 승인 완료; 구현 계획 작성 중.

동결 제안은 `docs/evaluation/p2-design-proposal-v1.md`이며 SHA-256은 `b312332c5864f64d09fe6d6be622e48c8a6fe4bc80b224bd56fb05f0a1cecfcf`다. 상세 비평과 source inspection은 `docs/evaluation/p2-design-review.md`에 기록했다. 제안과 아래 구현 규칙은 같다.

understood as: 완료된 P1 오프라인 VSP 평가기를 실제 원본 데이터와 기존 ILVR 추론 함수에 연결한다. 데이터는 활성 저장소의 `data/`에 두고 기존 symlink를 보존한다. CoMT, VisuLogic, EMMA, Zebra 2D와 최종 통합 검증은 후속 단계로 유지한다.

## 1. 선행 승인과 범위

P1의 schema, 엄격한 `vsp_validated_v1` scorer, manifest 분모 및 실패 계약은 승인·구현되어 있다. 기존 worktree `.worktrees/evaluation-p1`에서 이어가며 자동 commit/merge/push하지 않는다. 상위 문서는 `documents/migration_plan_codex_reference_ko.md`, `docs/evaluation/audit.md`, P1 승인 기록 `docs/superpowers/specs/2026-09-10-ilvr-eval-foundation-vsp-design.md`다.

2026-09-10 사용자는 설명 그림·예시 그림·문제 그림 3장을 기존 ILVR 경로로 전달하는 입력 제안에 `계속`으로 응답했다. 이 응답은 입력 방향 승인으로 기록하며, 이 문서의 최종 구현 승인으로 확대하지 않는다. P1에서 승인한 저장 위치·채점·실패 계약은 재사용한다.

접근 선택:

- 선택: 기존 `eval.py::load_evaluation_artifacts`와 `run_one_example`을 지연 import하는 얇은 bridge. 이미 반환하는 생성 suffix와 prompt identity를 재사용한다.
- 대안: 새 추론 runner에 latent generation을 복제한다. 기존 경로와 별개로 동작이 달라질 위험과 유지 비용이 커 선택하지 않는다.
- 대안: 기존 prompt 인터페이스를 확장해 원본 Gemini의 text/image 교차 배치를 재현한다. 기존 ILVR 입력 경로 보존이라는 승인 방향과 달라 선택하지 않는다.

이번 범위는 VSP prepare CLI, manifest 전용 preflight 재사용, inference bridge, run CLI, 실제 데이터 config, 해당 테스트와 실행 문서다. 기존 eval CLI, latent controller, patched Transformers, P1 score 규칙은 변경하지 않는다. 공유 manifest 검증 함수의 추출은 P1의 입력·오류 의미를 보존한다.

## 2. 원본과 데이터 준비

원본은 Visual-Spatial-Planning commit `6d74d7bb46ae71bf41aa4d3d874a4f31fc5f0366`의 `VSP-main/maze/task-main`이다. 원본 저장소는 읽기 전용이며 실행하거나 dependency로 설치/import하지 않는다. 실제 Git root/origin/HEAD/clean 및 기존 lock의 근거·license hash를 확인한 후 읽는다. 새로 사용하는 prompt 텍스트와 두 공통 그림의 hash도 lock에 추가한다. 원본 텍스트·그림을 복사할 때 해당 MIT license와 원본 경로를 데이터 출처 문서에 포함한다.

`prepare_eval`은 원본 Git checkout을 입력으로 받아 `maps/level3`부터 `level8`까지의 `table/<id>.txt`, `pure_text/<id>.txt`, `img/<id>.png`를 basename으로 대응시킨다. 각 level에서 정확히 0–99의 세 파일 집합을 요구한다. table의 `_/#/@/*`는 각각 `0/-1/1/2`로 변환한다. 크기, 허용 셀, 시작/목표 개수와 pure_text의 지도 크기·모든 좌표·구멍 없음 표기를 검증한다. 불일치와 누락을 건너뛰지 않는다. PNG를 정상적으로 읽을 수 있는지도 확인한다.

새 데이터 디렉터리는 `data/evaluation/<dataset-id>/`이다. 원본 문제 그림·table·pure_text는 `maps/levelN/{img,table,pure_text}/` 구조로, 두 공통 그림은 `prompt-visual-images/system-figure-{1,2}.png`로 보존한다. 원본 prompt는 `prompt-text/prompt-text.txt`, 가공 prompt는 `prompt-text/ilvr-three-images-v1.txt`에 둔다. 출처/license와 manifest도 이 데이터 디렉터리에 저장한다. 준비한 데이터는 Reference를 가리키는 symlink를 갖지 않는다. 기존 데이터 디렉터리는 덮어쓰지 않는다. 모든 원본 입력 검증을 복사 전에 마치고, 준비 중 실패하면 성공 manifest로 표시하지 않고 로그를 남긴다. 자동 정리·재시작 시스템은 만들지 않는다.

full manifest는 600행, 범주 `level3`…`level8`, category별 100행이다. 별도 smoke manifest는 level별 `0`번 1개씩 총 6행이다. smoke는 full manifest의 명시적 부분집합이며 전체 평가 점수로 보고하지 않는다. ID는 `vsp/levelN/<source-id>`이고 source-id는 원본 정수 basename의 문자열이다. full/smoke manifest는 동일한 샘플 ID와 같은 데이터 자산을 공유할 수 있으며 manifest byte hash는 다르다. provenance의 split 명칭은 `repository-maps`로 기록하고, 표본 선택은 manifest와 준비 보고서에 명시한다. 논문 test split이라는 주장은 하지 않는다.

prepare CLI 사용 계약:

```text
PYTHONPATH=src:. ./venv/ilvr-venv/bin/python -m scripts.prepare_eval \
  --benchmark vsp --profile vsp_validated_v1 \
  --source-root Reference/Visual-Spatial-Planning \
  --output-dir data/evaluation/<fresh-dataset-id>
```

결과는 `manifest.jsonl`, `smoke-manifest.jsonl`와 위 데이터 자산이다. 준비 요약은 `results/evaluation/<fresh-dataset-id>/prepare.json`, 로그는 `logs/evaluation/<fresh-dataset-id>/prepare.log`에 둔다. 세 목적지 모두 기존 경로와 충돌하지 않아야 한다. CLI가 원본 데이터를 읽는 것은 데이터 준비 단계에 한정되며 이후 run/score는 Reference 없이 실행한다.

## 3. 입력 프로토콜

순서는 1번 설명 그림, 2번 규칙 예시 그림, 3번 해당 문제 그림이다. 텍스트는 pinned `prompt-text/prompt-text.txt`를 기반으로 `<IMAGE-1>`, `<IMAGE-2>`, `<TEST-IMAGE>`를 각각 명시적 `Image 1 (legend)`, `Image 2 (worked example)`, `Image 3 (maze to solve)` 참조로 바꾸고, 앞에 그림 순서를 설명하는 한 문장을 둔다. 최소 이동 요청과 원본의 Action plan 출력 지시는 유지한다. 추가 few-shot 예시는 넣지 않는다.

기존 runtime은 모든 image placeholder 다음에 단일 text block을 넣어 user-only chat template을 적용한다. 그 경로를 그대로 사용한다. 모델에는 `text_input`과 해시 검증한 3개 이미지의 절대 경로만 전달한다. target의 map_desc, 좌표 sidecar, 정답 경로, 해설이나 gold label은 전달하지 않는다. 원본에 포함된 규칙 예시의 좌표는 문제별 gold가 아니므로 유지한다.

`prompt_template_identity`는 `vsp-ilvr-three-images-v1:<가공-prompt-UTF8-SHA256>` 형식으로 실제 가공 텍스트에서 계산한다. run preflight가 각 샘플의 텍스트가 이 가공 prompt와 같은지, config의 identity hash가 실제 텍스트 hash와 같은지 확인한다. 이미지 경로 순서는 위 두 공통 그림 다음 `maps/<subtask>/img/<source-id>.png`여야 하며 세 파일의 hash도 P1 규칙대로 검사한다. P1 scorer는 계속 임의의 유효한 VSP manifest를 받을 수 있으며 이 추가 제한은 P2의 VSP 생성 경로에만 적용한다.

결과 프로토콜은 `modified`다. 이 이름은 이미지 배치 변경과 P1의 엄격한 parser를 모두 포함한다. 원본 환경의 일부 규칙을 채택했다는 이유로 Gemini 프로토콜이나 논문 재현을 주장하지 않는다.

## 4. 실행과 파일 계약

추가 파일 후보는 `src/evaluation/vsp_prepare.py`, `src/evaluation/inference_bridge.py`, `scripts/prepare_eval.py`, `scripts/run_eval.py`, `configs/eval/vsp_ilvr_three_images_v1.yaml` 및 이에 대응하는 테스트다. 실제 함수 signature는 최종 설계 승인 후 계획에서 고정한다. 작은 공통 IO 함수 추출은 `src/evaluation/io.py` 안에서 수행한다.

run CLI는 기존 P1 EvalConfig 형식을 사용한다. config에는 실제 dataset_root, 6개 category, max_steps=100, 선언한 model_identity, 위 prompt_template_identity, 명시적 generation_config가 필요하다. run fingerprint는 P1과 동일하게 model_identity/prompt_template_identity/generation_config 세 필드의 canonical JSON SHA-256이다. manifest bytes의 SHA-256은 별도 필드다. 따라서 fingerprint를 모델 weight 파일의 내용 증명으로 설명하지 않는다. 예제 config의 미설정 식별자를 실제 실행 가능하다고 설명하지 않는다. 실행 문서에 데이터 준비 후 값을 채우는 정확한 절차를 제공한다.

```text
PYTHONPATH=src:. ./venv/ilvr-venv/bin/python -m scripts.run_eval \
  --manifest data/evaluation/<dataset-id>/smoke-manifest.jsonl \
  --config configs/eval/<configured-run>.yaml \
  --model-dir checkpoints/<artifact> \
  --model-identity <same-identity-as-config> \
  --output-dir results/evaluation/<fresh-generation-run>
```

PEFT에 필요한 `--base-model-dir`은 기존 loader 계약에 따라 전달한다. 선택적인 `--cache-dir`의 기본값은 `data/cache/evaluation`이며 data 밖을 거부한다. 기존 load 함수가 받는 cache_dir 인자를 그대로 전달한다. 이미 다운로드된 모델을 임의로 변환하거나 다른 경로에 복사하지 않는다.

실행 순서는 다음과 같다.

1. fresh results/log 경로, config의 benchmark/profile, manifest의 전체 ID·지도·분모·이미지 존재/hash와 이미지 디코딩, VSP 입력 프로토콜, 생성 설정, 모델/base 경로와 cache 경로를 검사한다. 이미지 검사는 Pillow를 사용해도 되지만 torch/Transformers/legacy 모듈은 아직 import하지 않는다. 모델과 base는 checkpoints 아래 존재하는 로컬 디렉터리로 제한한다.
2. `--model-identity`와 config의 선언이 같은지 확인한다. 이 식별자는 사용자가 지정하는 artifact identity이며 weight content hash가 아니다. 실제 model/base 경로와 작은 config/processor metadata hash를 run 기록에 남긴다. 동일 경로의 모델 파일을 실행 중 변경하지 않는 것을 전제로 한다. resume나 여러 실행의 자동 병합은 지원하지 않는다.
3. generation_config의 필수 키는 `max_new_tokens`, `max_sequence_length`, `repetition_penalty`, `temperature`, `top_p`, `do_sample`이다. 기존 CLI와 같은 형식·범위를 검사하고 `do_sample == (temperature > 0)`을 요구한다. P2는 이 여섯 키만 지원한다. kwargs는 값 변경 없이 기존 함수로 전달한다.
4. 입력이 유효하면 run 기록과 빈 predictions/generation-details 파일을 만들고, 그 뒤에만 legacy loader를 import하여 모델과 processor를 로드한다. load 실패도 빈 prediction 파일과 실패 기록을 남겨 P1에서 모든 예측 누락으로 재채점할 수 있다. 기존 loader의 Stage 1/2 및 full/PEFT 분기를 재사용한다. checkpoint의 stage, latent 크기·토큰 ID, 실제 모델/generation 설정, processor/chat template, Transformers 위치·버전을 기록한다. 선언된 클래스 존재만으로 어떤 controller가 실행되었다고 기록하지 않는다.
5. manifest 순서대로 batch=1의 `run_one_example`을 호출한다. bridge의 생성 인터페이스에는 ModelInput, dataset root, 모델/processor와 kwargs를 전달하며 Target은 전달하지 않는다. 모델 함수가 받는 sample에는 text_input/image_input만 있다. ID와 target은 CLI의 기록·채점 계층에 남는다. P1 Prediction의 response는 반환한 `assistant_only`, prompt_identity는 반환값, termination_reason은 stop_reason이다. prompt 전체를 다시 파싱하거나 `extracted_final_answer`를 response로 쓰지 않는다.
6. 생성마다 `generation-details.jsonl`에 동일 id/manifest hash/run fingerprint와 generated_suffix_token_ids, prompt_token_length, stop_reason, latent counts, timing을 먼저 저장한 다음 `predictions.jsonl`에 P1 Prediction을 저장한다. `raw_output`은 진단 sidecar에 보존할 수 있으며 scorer 입력과 섞지 않는다. 생성 상세 저장 실패는 실행 실패이며 완료 상태를 쓰지 않는다.

샘플 inference 예외는 해당 ID의 generation-details와 P1 Prediction 양쪽에 status=infrastructure_error와 error를 남기고 중단한다. 얻지 못한 suffix/prompt/종료 정보는 null이며 P1 response도 null이다. 나머지 ID는 missing이다. 자동 재시도하지 않는다. 모델 load 실패는 실행 오류로 기록한다. 빈 응답·문법 실패·token limit는 정상 관측 응답이며 bridge가 오답 판단이나 재생성을 하지 않는다. 기존 controller가 반환하지 않는 종료 의미를 임의 추정하지 않는다.

새 run 디렉터리에는 predictions, generation-details, run metadata와 generation 완료 상태를 저장한다. 로그는 같은 run-id의 `logs/evaluation/`에 저장한다. 생성 CLI는 benchmark accuracy를 계산하지 않는다. P1 score CLI를 별도의 fresh score run-id로 호출한다. 생성 완료는 모든 manifest ID의 두 출력 레코드가 정상 보존된 경우에 한하며 정답률과 독립적이다. 종료 코드는 입력 오류 2, 실행/출력 오류 1, 전체 생성 완료 0이다.

## 5. 검증과 완료 범위

CPU 계약 검증과 legacy를 로드하는 stub 테스트를 분리한다. 새로운 offline scorer 테스트는 torch/CUDA/model/Reference를 요구하지 않는다. bridge stub 테스트는 필요한 기존 torch 런타임을 쓸 수 있지만 다운로드나 실제 weight load를 하지 않는다.

- prepare: 허용 기호·좌표·no-hole·ID와 파일 대응, 불일치/누락 거부, 3장 순서, source copy와 hash, fresh 경로, target가 model_input에 섞이지 않음을 검증한다. 작은 독립 fixture를 쓰며 600개 source 검사는 별도 데이터 증거다.
- preflight: 유효하지 않은 manifest/profile/asset/kwargs/path에서 legacy import와 loader 호출이 0회임을 runtime witness로 확인한다. 기존 P1 계약 테스트로 공통 IO 추출의 회귀를 확인한다.
- bridge: 동일한 stub model/processor/controller와 3개 서로 다른 이미지로 기존 `run_one_example` 직접 호출과 bridge 호출을 비교한다. 실제 messages, tensor 입력/이미지 순서, generation kwargs, prompt token boundary, suffix IDs와 response를 확인한다. gold만 바꾼 입력의 메시지·생성이 바뀌지 않는지도 확인한다. 이를 GPU 수치 동등성이라고 보고하지 않는다.
- CLI integration: fresh data/results/log 아래 작은 manifest를 준비해 stub generation 후 실제 score CLI로 재채점한다. Reference 경로가 없는 환경에서도 prepared data의 run/score가 동작함을 확인한다. 실패/빈 응답/token-limit의 분모와 exit code를 검사한다.
- 실제 데이터: 준비한 full 600행과 smoke 6행의 ID·category·hash·좌표 대응을 검증한다. 합성 응답을 full benchmark 모델 성적으로 보고하지 않는다.
- GPU smoke: 사용 가능한 실제 checkpoint와 GPU 자원을 확인한 뒤 6개 범주별 1문제에서 직접 함수와 bridge를 같은 환경·입력·설정으로 비교한다. do_sample=false, temperature=0, 고정 seed로 시작하고 두 호출 전에 Python/NumPy/torch 난수 상태를 맞춘다. 기존 함수가 호출마다 새 controller를 생성하는 경로를 사용하며 KV cache/latent state 초기화 경로도 확인한다. 상태 격리나 초기화를 확인하지 못하면 비교 완료로 표시하지 않는다. 변경된 generation 코드를 따로 만들어 비교하지 않는다. 생성 차이는 그대로 기록하고 원인 확인 전 parity pass로 처리하지 않는다. 모델 경로를 찾지 못하거나 자원이 없으면 CPU 구현을 계속하고 GPU는 not_run으로 남긴다.

원본 Gym 실행, full 600개 모델 평가, 논문 split/프로토콜 검증은 이 P2의 CPU 구현 완료와 별도 상태다. 현재 Gym은 미설치이며 dependency 설치는 승인 범위에 넣지 않는다. GPU 확인 없이 P2의 모든 검증이 완료됐다고 쓰지 않는다. 기존 전체 프로젝트 테스트나 학습 동등성도 이 증거에서 추론하지 않는다.

## 6. 이미 확보한 근거와 한계

`results/evaluation/p2-discovery-20260910-01/source-discovery.json`은 600개 map의 basename 집합, table/pure_text 좌표 일치, PNG integrity를 기록한다. 이미지 자체의 의미는 level별 0번 총 6장을 별도 육안 대조했으며 나머지 594장의 의미 일치까지 증명하지 않는다. 두 텍스트 형식은 같은 생성기를 공유했을 수 있다.

`logs/evaluation/p2-discovery-20260910-01/existing-runtime-tests.log`의 4 passed는 기존 stub runtime 테스트이며 새 bridge의 테스트가 아니다. 로컬 TRAIN/TEST의 모든 이미지 경로는 images_comt 아래에 있고 source ID/지도 metadata가 없어 VSP에 사용하지 않는다.

과거 평가 기록에 적힌 `checkpoints/evaluation-staging/eval-origin-ilvr-stage1-dora-b1-e1-lr2p5e5-egb8-20260902-171710/origin-full-model`은 이번 확인에서 존재하지 않았다. 대신 그 기록의 source adapter `checkpoints/evaluation-staging/eval-stage1-dora-b1-e1-lr2p5e5-egb8-continuous-v29-20260902-122700/local-model`과 base `checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct` 디렉터리·metadata는 존재한다. adapter metadata는 Stage 1, latent_size=8, DoRA를 명시한다. 이를 우선 smoke 후보로 삼되 실제 weight 완전성·loader 호환성과 GPU 자원은 실행 전에 확인한다. 과거 성공 로그나 metadata만으로 지금 로딩 가능하다고 가정하지 않는다.

## 7. 비평 합성과 위험 수용 요청

**합의된 문제·범위·최종 설계:** 1–5절의 VSP prepare → 기존 ILVR bridge → P1 score 연결이다. 세 그림의 순서와 기존 추론 함수는 보존하고, 배치 변화와 strict scorer를 modified 프로토콜로 명시한다. P1 계약과 데이터/results/log/checkpoint 저장 정책을 유지한다.

**제안 대비 변경:** 동결 제안 v1의 scope, interface, behavior 변경은 없다. cold read의 GPU 난수/상태 통제, 실패 레코드 위치, fingerprint 의미 지적은 동결 전에 반영했다. 아래는 새 구현 규칙이 아니라 비평과 근거 수준의 합성이다.

**hate root:** 원본 PNG와 target 지도 대응이 틀리면 정확한 bridge/scorer라도 보지 않은 지도로 채점한다. **first_nail:** 지도 크기별 원본 PNG 한 장을 직접 읽어 같은 basename의 좌표 metadata와 대조한다. **상태와 관찰:** 읽기 전용 source inspection으로 level3–level8의 0번 총 6장을 확인했고 반례를 찾지 못했다. 모델 실행이나 prepare 코드 실험은 하지 않았다. 선택이 무작위가 아니고 나머지 594장의 그림 의미는 검증하지 않았다.

**macrothink:** 입력 방향이 명시적으로 승인된 기존 P1의 후속 연결이고 새 모델/훈련/평가 방법 선택이 없어, 추가 fresh reads 비용을 정당화할 wrong-problem 위험이 없다고 판단하여 생략했다. 방향이 독립 검증되었다는 뜻은 아니다.

**prism의 차이:** 기존 함수 위임과 비교 테스트를 요구하는 인터페이스 설계는 받아들일 수 있다. 반면 원본 이미지 의미의 전수 대응·GPU 동등성·논문 프로토콜까지 준비됐다는 주장은 unclear다. 이 판단 차이는 별도 상태 보고와 아래 risk waiver 요청으로 명시하며 통과로 합치지 않는다.

**mandela:** table/pure_text 두 형식의 일치만으로 그림 의미를 독립 검증한 것은 아니다. raw PNG 육안 대조 6장을 추가 근거로 갖지만 전체 자료에 대한 독립성 보증은 아니다. CPU fixture 점수와 실제 모델 성적은 계속 구분한다.

**미검증 전제:** 새 bridge의 실제 호출·입력·suffix 경계, P1 공통 IO 추출의 회귀 여부, 원본 594개 PNG의 의미 대응, 후보 adapter/base의 실제 로딩과 GPU 자원, GPU 상태 통제·생성 동등성, Gym runtime 대조, full benchmark와 논문 프로토콜이다. 향후 테스트와 실제 실행으로 확인할 부분을 현재 pass로 기록하지 않는다.

**명시적 위험 수용 — 2026-09-10 승인 완료:** 이 문서의 modified 프로토콜로 P2 데이터 준비와 inference 연결을 구현하고, 범주별 GPU smoke는 명시한 후보 모델의 사전 조건이 충족될 때 수행한다. 원본 그림 의미는 6/600까지 직접 대조했으며 그 밖의 한계와 미실행 검증을 그대로 공개한다. CPU 구현만 끝났을 때 GPU/전체 benchmark 완료를 주장하지 않는다. 새로운 dependency, 학습, 외부 judge, 자동 merge/push는 승인 범위에 포함하지 않는다.

## 8. 승인 후 경로

최종 commitment 승인에 따라 `superpowers:writing-plans`로 구체적인 실패 테스트·파일/함수·검증 명령을 정한다. 구현은 이 설계와 계획, TDD/필요한 review를 통해 진행한다. 계획 시작 시점에는 production P2 코드·config·준비 데이터·GPU 실행이 없다.

승인 기록: assistant가 최종 설계 v1과 명시한 검증 한계를 승인하고 구현 계획 단계로 진행할지 물었고, 사용자가 `ㅇㅋ`라고 응답했다. 입력 방향 승인과 별개의 최종 설계 승인이다. AGENTS.md의 구현 경로 선택에 따라 현재 세션에서 subagent-driven-development를 사용한다.

## 9. 실행 결과 — 2026-09-10

승인된 P2 구현과 리뷰를 완료했다. 실제 VSP 데이터 600개 및 smoke manifest
6개를 data 아래 준비했고, 최종 CPU 계약 212개와 별도 torch 스텁 7개가 통과했다.
Reference 7개와 기존 runtime/Transformers source, 저장 링크 보존도 확인했다.
검증 driver의 임시 경로 부모 누락은 실패 로그를 남기고 경로 생성만 수정한 뒤
스텁 검증을 새 run ID로 완료했다.

H100 NVL의 기존 Stage1 checkpoint로 직접 함수/bridge 6쌍의 다섯 출력 필드가
일치했다. 1,460회 관찰된 Qwen forward는 cache 미사용이었으며, 이는 모든
내부 상태 초기화의 일반 증명은 아니다. 실제 run/score CLI는 각각 exit 0,
smoke 점수는 0/6이었다. 모두 128-token 제한에서 종료한 invalid_format으로
기록했다. 기본 4096-token 성능이나 full benchmark 점수로 해석하지 않는다.

기존 loader는 base checkpoint 경로 문자열을 정확히 비교하므로 canonical
경로는 containment/metadata에, 사용자가 지정한 lexical absolute 경로는 loader
인자에 사용했다. 첫 로드 실패와 새 run의 성공 기록을 보존했고 기존 loader나
checkpoint metadata는 수정하지 않았다. 전체 이미지 의미, Gym 대조, full
benchmark, paper protocol의 기존 한계는 남는다. 자세한 명령과 증거는
[validation.md](../../evaluation/validation.md)의 P2 최종 검증 절에 있다.
이 실행 결과는 승인 당시의 6–8절 기록을 대체하지 않는다.
