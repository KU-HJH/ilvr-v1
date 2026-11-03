# design commitment v1 — P6 Zebra-CoT 2D 오프라인 평가

2026-09-14. 상태: **design commitment v1 승인 완료; P6 오프라인 구현·고정 400개 준비·CPU 검증 완료**.

understood as: 사용자가 선택한 Jigsaw/Search 각 200개에 대해, 기존 오프라인 평가 경로를 확장한다. Search는 대소문자·공백·문장 끝 마침표만 정규화해 전체 답을 비교한다. `red`와 `The bicycle is red.`가 불일치한다는 한계를 제시한 뒤 사용자가 이 비교 범위를 선택했다. 아래 완료 상태는 오프라인 구현·데이터 준비에 한하며 모델 성능·논문 재현 완료를 뜻하지 않는다.

## 1. 범위와 승인된 선택

`benchmark=zebra_2d`, `profile=zebra_2d_rules_v1`을 명시한다. 과제는 `Jigsaw`, `Search`다. 새 기본값이나 unknown fallback은 없다. 기존 prepare/score CLI, Sample/Prediction, manifest hash, run fingerprint, 경로 검사, 누락·인프라 오류 처리와 과제별 집계를 재사용한다.

추가 파일은 `src/evaluation/benchmarks/zebra_2d.py`, `src/evaluation/zebra_2d_prepare.py`, `configs/eval/zebra_2d_rules_v1.yaml` 및 Zebra CPU 테스트다. `schema.py`, `io.py`, `registry.py`, prepare CLI에는 Zebra 분기만 추가한다. 보고서에는 Zebra 한정 프로토콜 설명을 추가한다. 기존 benchmark의 target, 추출 규칙, 결과 형식과 CLI 사용법은 유지한다.

기존 `eval.py`, `src/evaluate_deepseed.py`, `src/task_deepseed.py`, `evaluation/{scoring,runtime,continuous_latent}.py`, patched Transformers, P2 bridge와 `run_eval.py`, prompts·이미지 순서·생성 설정은 변경하지 않는다. GPU 추론, 모델 준비, judge 실행과 의미 비교 확장은 이번 범위가 아니다. `Reference/`는 읽기 전용 근거이며 runtime에서 import/install하지 않는다. 새 의존성·재시도·복구 프레임워크를 추가하지 않는다.

## 2. 고정 원본과 준비 계약

원본 dataset은 `multimodal-reasoning-lab/Zebra-CoT`, revision은 `f6d1defc169d19180a69f517925ed6cbb06a1c97`이다. config는 `2D Visual Reasoning - Visual Jigsaw`와 `2D Visual Reasoning - Visual Search`, split은 모두 `train`이다. 53개 footer의 51,899행을 확인한 뒤 정답과 무관한 SHA-256 순서로 각 200개를 고정했다. 표본 추가·제외·교체·정답 수정은 없다.

입력은 `data/evaluation/zebra-source400-20260914-02/`의 원본 투영 자료다. 준비에 필요한 파일은 `selected-ids.json`, `source-records.jsonl`과 그 안에 열거된 400개 이미지다. 준비기는 선택 파일 SHA-256 `92e776e6b7e7e646c7f3b23c05c87b0993af44821faecac789a0e29ef2737927`, 원문 파일 SHA-256 `1a533ce69f426cce9397251df976cf4331118f530b342713747f2ebdae3d4f25`를 코드의 고정 값과 대조하고, 400개 고유 ID·과제별 200개·두 파일의 행별 대응·각 이미지 hash/디코딩을 검사한다. 이 digest들은 수집한 투영 자료를 고정하며 원본 parquet 전체 bytes의 검증은 아니다. 운영 준비기가 원격 다운로드하거나 원본 53개 parquet를 요구하지 않는다.

원문과 선택 정보의 타입·필수 필드·과제/config/split·전역/지역 행 번호는 저장된 계약에 맞아야 한다. 이미지는 소스 root 내부의 기존 일반 파일이며 경로 탈출을 거부한다. 잘못된 입력·누락/손상 이미지·digest 불일치는 준비 실패이며 다른 문항으로 대체하지 않는다. 기존 목적지를 덮어쓰지 않는다. 준비 전 전체 입력을 검사한 뒤 출력한다.

선정 JSON은 record 400개의 배열이다. 각 record의 정확한 키는 `task`, `config`, `split`, `global_row`, `shard_path`, `shard_row`, `selection_hash`다. 행 번호는 bool이 아닌 0 이상의 int이고 나머지는 비어 있지 않은 문자열이다. 원문 JSONL record는 이 7개 키에 문자열 `Question`, `Final Answer`와 객체 `problem_image_1`을 더한다. 이미지 객체의 정확한 키는 `path`, `source_path`, `sha256`, `bytes`, `format`, `width`, `height`, `mode`다. source_path는 원본 이미지 cell의 문자열/null 메타데이터이며 파일 접근에 사용하지 않는다. path/hash/format/mode는 비어 있지 않은 문자열, bytes/width/height는 bool이 아닌 양의 int다. 실제 400개는 JPEG이며 hash는 64자리 lowercase hex다. 이것은 위 digest로 고정된 투영 자료의 필드 목록이며 신규 수집 포맷을 일반화하지 않는다.

출력 dataset profile은 `zebra-2d400-v1`, 목적지는 `data/evaluation/zebra-2d400-v1/`다. 기존 prepare 패턴대로 `manifest.jsonl`, `source-selection.json`, `SOURCE.json`, 원본 bytes를 그대로 보존한 images를 쓴다. 선정 메타데이터의 shard path/row·global row·원문·정답과 자산 hash는 감사 자료로 보존한다. 준비 summary는 N_expected=400, Jigsaw/Search=200씩, image_count=400, excluded_rows=0, 원본 identity·profile·source digest·manifest digest를 포함한다.

`Sample.id`는 `zebra_2d/{subtask}/{global_row:05d}`, `Provenance.source_id`는 동일한 5자리 전역 행 번호다. config별 0-based 행 순서는 고정 shard 경로 사전순과 parquet 내부 순서다. 원본 config/split/revision과 이미지 hash는 기존 Provenance 필드에 기록한다.

`ModelInput.text`는 원본 `Question`을 변경 없이 보존하며 `<image_start>[problem_image_1]<image_end>`도 유지한다. `images`는 해당 문제 이미지 상대 경로 1개, `choices`는 빈 tuple이다. 이미지 안의 선택지를 gold 문자에서 추측해 텍스트로 만들지 않는다. `Final Answer`와 reasoning trace/image는 ModelInput에 넣지 않는다. 새로운 `ZebraTarget(answer: str)`에 원본 정답을 보관하며 공통 Sample target union에만 이를 추가한다.

공통 reader와 scorer 직접 호출 모두 subtask·target을 검사한다. Zebra의 ModelInput은 비어 있지 않은 text, 이미지 1개, 빈 choices를 요구한다. Jigsaw target은 대문자 A–D 한 글자다. Search target은 문자열이며 아래 정규화 후 비어 있지 않아야 한다. 다른 benchmark의 target 검증을 바꾸지 않는다. 일반 scorer용 manifest는 고정 400개 외의 명시적 Zebra fixture/부분 집합도 허용하지만, 운영 준비기는 고정 400개만 만든다.

## 3. 정답을 보지 않는 추출 문법

`extract_answer(response, model_input)`은 target·정답·원본 답안 목록을 받지 않는다. 공통 문자열 후보를 추출한 뒤 Jigsaw의 라벨 문법 검사는 subtask만으로 수행한다. 비교 단계만 target을 읽는다. 아래 규칙은 기존 EMMA/VSP 추출기를 바꾸지 않는 Zebra 전용 계약이다.

호출 경계는 `score_zebra_2d(sample, prediction)`이 소유한다. 이 함수가 target/입력 계약을 검사한 뒤, response와 ModelInput만 추출기에 전달한다. 반환된 후보에 대해 `sample.subtask`가 Jigsaw이면 정답을 받지 않는 라벨 문법 검사를 호출하고, Search이면 아래 문자열 정규화의 빈 결과 여부를 검사한다. 검사 결과는 ParsedAnswer의 status/error에 반영하되 원래 후보 value와 method를 보존하고 나서 target과 비교한다. 추출기가 ModelInput에서 subtask를 추측하거나 target의 형태로 과제를 결정하지 않는다.

1. 응답 전체가 공백이면 `empty`다. 앞뒤 공백을 제거한 응답을 사용한다.
2. 정확한 토큰 `\boxed{`가 있으면 마지막 최상위 box의 내용을 택한다. 중괄호 깊이를 세며, 내부 box는 별도의 최상위 후보가 아니다. 마지막 최상위 box가 닫히지 않으면 `invalid_format`으로 끝나고 이전 box나 marker로 돌아가지 않는다. 중괄호 escape 해석은 지원하지 않으며 모든 `{`/`}`를 구조 문자로 센다.
3. 선택한 box 내용 전체를 감싼 완전한 `\boxed{...}` 또는 `\text{...}` wrapper만 반복해서 벗긴다. 주변 공백은 제거한다. 중간의 wrapper나 `$...$`, `\(...\)`는 해석하지 않는다. box 밖의 문장은 후보에 붙이지 않는다.
4. box가 없으면 줄 시작의 마지막 answer marker를 찾는다. 줄 시작에는 ASCII space/tab을 허용하고 대소문자를 무시한다. 형태는 `Final answer:`, `Final answer is:`, `Final answer is `, `Answer:`이며 선택적 선행 `The `를 허용한다. 단어 사이는 ASCII space/tab 1개 이상, colon은 `:` 또는 `：`다. `is` 뒤에 colon이 없으면 space/tab이 필요하다. marker 뒤의 **전체 나머지 응답**을 후보로 삼는다. 첫 줄·첫 마침표에서 자르지 않는다.
5. marker도 없으면 응답 전체를 후보로 삼는다. 비어 있는 box/marker 후보는 `invalid_format`이다. 답으로 보이는 단어를 본문 중간에서 찾거나 gold를 보고 후보를 다시 선택하지 않는다.

ParsedAnswer는 정규화 전 후보, 기존 허용 method 중 `raw`/`boxed`/`marker`, status와 실패 이유를 기록한다. malformed box, 빈 후보, Jigsaw 라벨 문법 실패를 구분한다. 임의 모델 문자열을 코드로 실행하지 않는다.

## 4. 과제별 비교 규칙

**Jigsaw:** 앞뒤 공백 제거 후 A–D/a–d 한 글자 또는 그 한 글자를 ASCII 괄호로 감싼 `(A)` 형태를 허용한다. 각 형태 끝의 ASCII 마침표 1개를 허용한다. `( A )`, 여러 라벨, `Option B`, `B: 설명`, `B because...`는 허용하지 않는다. 라벨만 대문자로 바꿔 target과 비교한다. 빈/잘못된 문법은 모델 실패로 분모에 포함하며 정답 선택지 내용으로 재해석하지 않는다.

**Search:** 후보와 원본 target 각각에 `str.casefold()`, Unicode 공백 단위 분리 후 ASCII space 하나로 재결합을 적용한다. 마지막 문자가 ASCII `.`이고 그 앞 문자가 `.`이 아닐 때만 마침표 하나를 제거하고 뒤 공백을 제거한다. 따라서 `word.`와 `word`는 같지만 `word..`는 보존한다. 이 순서 뒤 비어 있는 후보는 `invalid_format`이며 빈 문자열끼리 정답 처리하지 않는다. 비어 있지 않은 두 결과 전체가 같을 때만 정답이다. target에는 answer marker/boxed 추출을 실행하지 않는다.

Unicode compatibility 정규화, 따옴표/괄호/관사 제거, 동의어·부분 문자열 비교, 숫자·분수·날짜·단위 변환, 수학 wrapper 해제는 없다. Search의 `A`/`B`는 문자 그대로의 답이며 yes/no alias가 아니다. `nan`/`inf`도 숫자로 실행하거나 특별 정답으로 바꾸지 않는다.

계약 예: `Knife.`와 `knife`는 일치한다. `A knife.`와 `knife`, `red`와 `The bicycle is red.`, `1/2`와 `0.5`, `9/25/87`와 `1987-09-25`, `$ 11 billion`과 `$11 billion`은 불일치한다. `3.14`의 내부 소수점은 보존한다. 정답인 의미를 다른 문장으로 표현해도 틀릴 수 있다. 이는 사용자가 선택한 보수적 규칙 지표의 제한이다.

## 5. 보고서·오류·실행 경계

config categories는 Jigsaw/Search의 비어 있지 않은 중복 없는 부분 집합이다. 기본 config에는 둘 다 넣고 max_steps는 null/미제공으로 둔다. 공통 검증이 config 밖의 Sample subtask를 거부한다. 준비와 scoring은 명시적인 benchmark/profile 조합만 받는다.

분모는 전체 예상 문항 수다. 빈/invalid 응답은 관측된 모델 실패다. prediction 누락·infrastructure_error는 미관측이며 해당 과제/전체 complete=false, accuracy=null이다. 완전한 평가는 correct/N_expected와 두 과제별 기존 count/observed/correct/complete/accuracy를 제공하고 전체 coverage를 유지한다. 빈 과제 count=0, accuracy=null과 기존 CLI exit 0/1/2 의미를 유지한다. 별도의 평균 방식이나 추가 의미 정답률을 만들지 않는다.

Zebra 보고서에만 `zebra_2d` 설명 객체를 추가해 `comparison_kind=conservative_rules`, `ood_verified=unknown`, `paper_sample_identity=unverified`, `paper_judge_equivalent=false`를 기록한다. 기존 top-level `protocol_kind=modified`, `verification.paper_protocol_verified=false`와 CPU/upstream/GPU/full_benchmark의 개별 확인 상태도 유지한다. config template의 model/prompt identity는 예시이며 실제 scoring에는 실제 prediction의 identity/settings를 명시해야 한다.

scorer와 공통 JSON 계약은 표준 라이브러리로 동작하고 YAML/Pillow는 이미 설치된 버전만 사용한다. 준비 시 이미지를 확인하는 Pillow를 제외하고 scorer가 이미지 파일을 디코딩하거나 네트워크/PyArrow/torch/CUDA/Reference를 요구하지 않는다. 모든 Python·tests·CLI는 활성 worktree의 `./venv/ilvr-venv/bin/`을 사용한다.

데이터·원본·manifest·이미지·데이터 fixture/cache는 활성 worktree의 `data/` symlink 아래에, 결과는 `results/evaluation/<run-id>/`, 로그는 `logs/evaluation/<run-id>/`에 둔다. `data/`·`checkpoints/` symlink를 보존한다. 모델 artifact를 만들지 않는다.

## 6. 구현 후 검증과 현재 근거의 한계

TDD로 box/marker 우선순위·중첩·미완성 box·소수점·복수 라벨·날짜/통화 보존·case/공백/마침표·관사/부분 문자열 거부·gold 교체 시 추출 불변을 검증한다. source 정답을 prediction으로 복사한 자기 일치를 모델 성능으로 보고하지 않는다.

준비/공통 reader 통합은 고정 파일 hash, 400개 고유 대응, 이미지 손상·경로 탈출·누락, 잘못된 category/target/choices, 입력에 target/helper를 전달하지 않는 실제 호출 경계를 검증한다. small fixture는 synthetic임을 표시하고 데이터 파일은 `data/` 아래에 만든다. `Reference/` 없는 격리 subprocess에서 torch/CUDA/모델 로딩 없이 새 scorer와 score CLI가 동작하는지 검사한다. 새 공통 분기는 P1–P5의 기존 scoped CPU 계약 테스트와 작은 CLI 통합 검증으로 회귀를 확인한다. 기존 runtime을 변경하지 않으므로 새로운 GPU 실행이나 반복 broad test를 추가하지 않는다.

실제 400개를 준비한 뒤, 구현 전에 고정한 원본 투영 자료와 **400개 모두의 ID·Question·target·이미지 bytes/hash**를 대조한다. 데이터 전수 검사, CPU scorer 계약, 원본 환경 비교, GPU smoke, 전체 모델 평가와 논문 프로토콜 검증을 따로 보고한다. P6 완료 주장은 오프라인 구현과 승인한 400개 준비 범위로 제한한다. 공개 51,899개 전체 평가 또는 논문 재현 완료로 확대하지 않는다.

현재 근거: `docs/evaluation/p6-zebra-discovery.md`, `results/evaluation/p6-source400-20260914-02/source-validation.json`. 실제 원본 400개·JPEG 400개·125,791,653 bytes를 확보했고 로컬 ID/hash/디코딩을 검사했다. Search의 날짜·금액·단어·178자 문장을 확인했다. 논문 평가 ID·학습 manifest·정확한 judge 계약은 확보하지 못했다. 고정 Bagel checkout에서 전용 scorer를 찾지 못한 기존 audit의 범위만 유지한다. 설계 승인 당시 신규 scorer 테스트·adapter·추론은 미실행이었다. 승인 후 오프라인 구현·검증 상태는 9절에 기록하며 추론은 계속 미실행이다.

## 7. 승인 이후 경로

사용자가 최종 명세 v1을 보고 `진행하자`로 승인했다. brainstorming을 완료하고 `superpowers:writing-plans`로 [상세 구현 계획](../plans/2026-09-14-ilvr-zebra-2d-offline-plan.md)을 작성했다. 기존 CLI의 실행 예와 파일별 실패 테스트/검증 명령은 그 계획에 고정했다. 사용자의 계속 진행 지시와 standing AGENTS의 실행 경로에 따라 이 세션에서 subagent-driven-development로 수행한다. 자동 commit/push하지 않고 기존 P1–P5 변경과 worktree를 보존한다.

## 8. 비평 합성 및 남은 확인

동결 `design proposal v1`은 `results/evaluation/p6-design-20260914-01/design-proposal-v1.md`, SHA-256은 `6b667f9292470311a66944f5de4e567105dabb7a0b7793c512ac07c6610750d0`이다. hate와 prism은 이 동일한 문서를 읽었다. 합성 과정에서 제안의 문제·범위·인터페이스·행동을 바꾸지 않았다. 기존 과제별 보고서에 coverage 필드가 있는 것으로 오해되지 않도록 이미 존재하는 필드 이름을 정확히 적었다.

**hate:** `{root: "gold와 분리된 추출이라는 선언만으로는 복잡한 응답에서 후보 선택이 일관되거나 target 교체에 불변임을 보장하지 못한다", first_nail: "구현 전 TDD에서 같은 raw/marker/중첩 boxed/미완성 boxed 응답을 서로 다른 target에 연결하고, 추출 결과 불변 및 계약상 후보를 실제 호출 경로로 확인한다"}`. 설계 검토 시 first_nail은 **not_run**이었다. 승인 뒤 TDD의 target 교체 사례와 추가 raw/marker/중첩 boxed/미완성 boxed 4분기·8회 실제 scorer 호출에서 동일 추출 결과를 관찰했다. [실행 근거](../../../results/evaluation/p6-first-nail-20260914-01/target-independence.json). 이는 검사한 합성 사례의 호출 경계 증거이며 모든 입력의 무누출 증명이나 실제 모델 의미 정확도 검증이 아니다.

**macrothink:** 실행하지 않았다. 사용자가 표본·비교 목적과 제한을 명시적으로 선택했고, 남은 작업은 기존 오프라인 경로의 P6 확장이다. 현재 잘못된 문제를 풀 위험에 대한 추가 fresh-read 비용보다 구체적 추출/입력 경계 검증이 유용하다고 판단했다. 이는 다른 방향이 틀렸다는 증거가 아니다.

**prism:** 같은 세션에서 서로 다른 두 실패 축으로 읽었다. 지표 의미 축은 **설계 수준 pass**: Search의 표기 정규화 점수를 의미 정답률/논문 judge 점수로 표시하지 않는다. 데이터 분리 축도 **설계 수준 pass**: 고정 digest가 묶는 source/selection과 단일 문제 이미지로 입력을 만들며 target은 비교에만 전달하도록 명시했다. 두 축은 서로 다른 이유로 동의했고 accept/reject를 바꾸는 이견은 발견하지 못했다. 이 합의는 독립 실행 증거나 코드 검증이 아니다. 판단을 바꿀 다음 증거는 위 first_nail과 준비 후 400개 전수 대조다.

**mandela:** 설계의 CPU 검증·데이터 검증·성능 주장 분리를 읽기 전용으로 점검했다. 원본 target은 외부 데이터에서 들어오며, 자체 fixture/target echo를 모델 성능으로 발표하는 경로는 제안하지 않았다. 현재 설계에서 확인된 leakage 패턴은 없지만 아직 없는 runtime의 무누출을 입증한 것은 아니다. 학습 manifest와 논문 평가 ID가 없으므로 OOD/표본 동일성을 검증된 값으로 바꾸지 않는다.

**이미 승인된 한계:** 사용자는 각 200개 구현 검증 표본이 논문 동일 표본/OOD로 확인되지 않았음을 전제로 선택했고, `red`와 문장형 gold가 불일치하는 사례를 제시한 후 보수적 비교 범위를 선택했다. 이를 다시 승인 요청하지 않는다.

**이번 최종 검토 범위:** 3절의 marker/boxed·실패 문법, 2절의 전용 target/준비 계약, 5절의 Zebra 보고서 정보와 6절의 검증·실행 경계다. 설계 승인 당시 신규 코드의 계약 준수·입력 분리·기존 경로 보존은 미검증이었다. 이 절의 설계 검토 자체는 테스트 통과 증거가 아니며 승인 후 실행 근거는 9절과 연결한 검증 기록에 있다. 원본 환경·GPU·실제 모델 평가·논문 프로토콜은 not_run/unknown으로 남긴다.

**shower:** 문서 복사본 전체를 대화/저장소 맥락 없는 별도 세션에 전달했다. 독해 결과는 **minor gaps**, 설계 blocker는 없었다. 과제별 문법 검사 호출 주체, 원본 필드 계약, 비평 합성을 미래 시제로 적은 문장이 불명확했다. scorer가 subtask 분기를 소유한다고 명시하고, 고정 투영 자료의 정확한 키/타입을 2절에 추가했으며, 7절을 현재 승인 상태에 맞췄다. 실행 명령의 부재는 설계와 후속 구현 계획의 경계로 명시했다. 의미/허용 입력/범위의 변경 없이 명시성을 보완했으며 새로운 제안 버전을 만들지 않았다. 독해 결과는 코드 검증 증거가 아니다.

## 9. 승인 후 완료 증거 — 2026-09-14

P6의 승인된 오프라인 scorer/reader/report/prepare CLI와 고정 400개 준비를
완료했다. Zebra 집중 83개, 최종 P1–P6 scoped CPU 633개가 통과했다. JPEG
pixel decode와 안전한 fixture 정리를 수정하고 한 번의 최종 수정 재검토를
통과했다. 수정 후 실제 400개 재준비 manifest는 기존 hash와 같고, 공통 reader가
기본 config의 기존 산출물 400행을 읽었다. ID·Question·target·이미지의 전수
대응 근거는 별도 보존했다. [최종 검증](../../evaluation/validation.md)에 명령,
실패/재실행 이력과 범위를 기록했다. 기존 파일 5,055개 및 Reference 7개/33개
hash를 보존했다. 모델 응답 평가·원본 환경·GPU·논문 프로토콜은 미실행이며,
OOD는 unknown이다. P7 통합 검증은 별도 후속 단계다.
