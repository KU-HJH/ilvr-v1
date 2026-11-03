# ILVR 평가 이식 P0 로컬 감사

감사일: 2026-09-10. 작업 루트: `/root/workspace/ILVR`.

understood as: 승인된 평가 이식의 다음 단계로 참조 코드와 현재 ILVR을 감사하고, P1 오프라인 계약/VSP 설계를 검토 가능하게 만든다. 평가 구현 소스는 수정하지 않는다. 모든 데이터는 현재 루트의 `data/`에 둔다.

## 확인 수준과 산출물

- 상위 명세: [migration_plan_codex_reference_ko.md](../../documents/migration_plan_codex_reference_ko.md).
- 코드 참조: [references.lock.json](../../references.lock.json). 실제 독립 Git root, origin, HEAD, clean 상태, 식별 ref, 라이선스 및 25개 근거 파일의 SHA-256을 기록했다.
- 환경/초기 변경 상태/소스 hash: [environment.json](../../logs/evaluation-p0-20260910-01/environment.json).
- 이전 기준과 현재 HEAD의 관련 diff: [baseline-diff.txt](../../logs/evaluation-p0-20260910-01/baseline-diff.txt).
- 현재 함수 조각 CPU 진단: [local-vsp-probe.json](../../results/evaluation-p0-20260910-01/local-vsp-probe.json).
- 기존 scorer 테스트: [tests.log](../../logs/evaluation-p0-scoring-20260910-02/tests.log), **11 passed**, exit 0. 새 evaluator의 테스트 결과가 아니다.
- 원본 Gym 환경, 실제 GPU 추론, 전체 benchmark, 논문 프로토콜 대조는 실행하지 않았다.

## 작업 환경

| 항목 | 실제 상태 |
|---|---|
| 현재 ILVR HEAD | `2497ee7bafad1b51ef6c2aba6f40f084830916c4` |
| 상위 명세의 이전 기준 | `13a67ed2ee975d6ca56b4e9eba18d68b352b166b` |
| branch / worktree | `main`, Git dir와 common dir 모두 `.git`; linked worktree가 아님. 변경하지 않음 |
| Python | `./venv/ilvr-venv/bin/python`, 3.11.15 |
| PyTorch / FlashAttention | `2.7.1+cu128` / `2.8.3` |
| Transformers | `4.50.0.dev0`, `/root/workspace/ILVR/transformers` editable 설치 |
| NumPy / MathRuler | `2.4.4` / `0.1.0` |
| Gym / Gymnasium | 둘 다 미설치 |
| pytest / PyYAML / SymPy / datasets | `9.1.1` / `6.0.3` / `1.14.0` / `4.2.0` |
| Codex CLI / Superpowers | `codex-cli 0.154.0` / 설치 경로의 버전 `6.3.0`; 플러그인 Git SHA는 주장하지 않음 |
| dependency 검사 | `pip check` exit 0, broken requirements 없음. 기존 pip 캐시 권한 경고는 로그에 보존 |
| 사용자 데이터 경로 | `data/` → `/raid/workspace/ILVR/data`, 기존 symlink 유지 |

초기 tracked 파일 변경은 없었고, 사용자 소유의 여러 untracked 문서·설정·캐시가 있었다. 정확한 목록은 환경 기록에 있다. 초기 index도 비어 있었다. P0에서는 dependencies, 브랜치, 학습/추론 소스, 원본 참조를 변경하지 않았다. `/Reference/`와 `/data`는 이미 Git 제외되어 있었다. 이 감사의 새 lock과 문서는 ILVR에서 추적할 대상이며 자동 commit/push하지 않는다.

## 참고 저장소

| 저장소 | 고정 SHA | 라이선스 파일 관찰 |
|---|---|---|
| Visual-Spatial-Planning | `6d74d7bb46ae71bf41aa4d3d874a4f31fc5f0366` | `LICENSE`: MIT |
| CoMT | `01095663a1d1f3b4818d255e0a8f02ef754bddc4` | tracked 파일 목록에서 LICENSE/NOTICE/COPYING/COPYRIGHT 파일 없음 |
| EMMA | `47259952dd8e95bc1e3855a4164ccd3634f8eb1e` | tracked 파일 목록에서 LICENSE/NOTICE/COPYING/COPYRIGHT 파일 없음 |
| VisuLogic-Eval | `00fba6ddafdc0d6805a4b9455c8df36502aeda2a` | `LICENSE`: Apache-2.0 |
| VLMEvalKit | `b06e0716f22127c1920364c6f4d80b9c32e42c09` | `LICENSE`: Apache-2.0 |
| Bagel-Zebra-CoT | `bbf53c327889a1764d27989d117d1a6b73ffdd3f` | `LICENSE`: Apache-2.0 |
| MathRuler | `1cf48eb78b244e3e231565bcfb184f2b294e3d3f` | `LICENSE`: Apache-2.0; `v0.1.0` commit 일치 |

7개 모두 독립 root/origin이 맞고, detached HEAD이며 tracked/untracked 수정이 없었다. `source_ref`는 로컬에서 확인한 식별 ref이고 실행 기준은 40자리 commit이다. CoMT/EMMA의 라이선스 부재는 관찰 범위에 대한 기록이지 외부에 어떤 허가도 없다는 법률 판단이 아니다. 허가 확인 전 소스 복사를 보류한다. P0에는 이식한 소스가 없으므로 `THIRD_PARTY_NOTICES.md`에 이식 완료 항목을 만들지 않았다.

## VSP: 현재 호출 경로와 진단

두 실행 진입점의 정적 호출 경로:

```text
main()
  -> run_one_example(...)
  -> pred["assistant_only"]
  -> task_name == "vsp-spatial-planning-cot" 분기
  -> sample.get("map_desc", [])
  -> extract_path_from_text(assistant_only)
  -> simulate_vsp(map_desc, path_str)
  -> parse_action_sequence(path_str)
  -> sim["success"] 집계
```

근거: [eval.py](../../eval.py:553), [evaluate_deepseed.py](../../src/evaluate_deepseed.py:342). 다른 task 이름은 `score_answer()` 경로를 탄다. 이름 등록만으로 VSP 지원을 판정하면 안 된다. [task_deepseed.py](../../src/task_deepseed.py:198)에도 복제 함수가 있다. `src/`, `eval.py`, `tests/` 검색에서 그 복제 함수를 명시적으로 부르는 지점은 발견하지 못했다. `src/main.py`의 wildcard import는 존재하며, 이 검색만으로 모든 동적 호출이 없다고 증명하지는 않는다.

`systematic-debugging`으로 현재 함수 정의와 `ACTION_MAP`만 AST에서 추출해 메모리에서 실행했다. CLI/module 전체를 import하지 않았으며 원본 Reference를 런타임 import하지 않았다. 추출기에는 **현재 venv에 설치된** MathRuler를 제공했다. 설치된 `grader.py`는 pinned Reference 파일과 바이트 hash가 일치했다.

| 입력 | 관찰 결과 | 범위 |
|---|---|---|
| 지도 `[[1,2],[0,0]]`, `R` | 성공 | 현재 시뮬레이터 3개 |
| 같은 지도, `RD` | 실패 | 목표에서 종료하지 않고 다음 이동 후 최종 셀 판정 |
| 같은 지도, `XR` / `RX` | 둘 다 성공 | parser가 X를 삭제해 `R`로 만듦 |
| `final answer: R` | 문자열 `"None"`, truthy | 두 현재 extractor + 설치된 MathRuler |

총 **14개 관찰**이다. 회귀 테스트 14개 또는 전체 진입점 동등성 검증으로 보고하지 않는다. `"None"`은 `extract_boxed_content()`의 실패 sentinel이며 현재 `if s:`가 참이 되어 fallback을 막는다. MathRuler가 없을 때와도 실행 의미가 달라진다. 다음 P1 테스트에서 이 관찰 사례를 새로 고정해야 한다.

원본 `VSP-main/maze/task-main/eval.py`는 `gym.make('FrozenLake-v1', desc=rows, map_name='4x4', is_slippery=False)` 후 `terminated or truncated`에서 멈추며 reward가 양수일 때 정답을 센다. parser는 `Action plan:` / `Action Plan:` 뒤를 comma로 분리하고 따옴표·줄바꿈·마침표를 제거한다. 따라서 strict parser는 원본과 구별되는 수정 profile이다. prompt에는 최소 이동 요구가 있으나 이 evaluator에서 shortest-path 검사를 찾지 못했다.

원본 환경 파일은 루트의 `gemini.yaml`이 아니라 **`VSP-main/gemini.yml`**이며 Python 3.12.2 / Gym 0.26.2 / NumPy 1.26.4를 명시한다. Gym 0.26.2의 [FrozenLake 등록](https://raw.githubusercontent.com/openai/gym/0.26.2/gym/envs/__init__.py)은 100 step, [TimeLimit.step](https://raw.githubusercontent.com/openai/gym/0.26.2/gym/wrappers/time_limit.py)은 전이 후 elapsed step을 올려 `>= limit`에서 truncated를 설정한다. 이는 소스 확인이며 실제 wrapper 실행 증거가 아니다.

원본 evaluator의 상대 경로 `../maps/level5_text/`는 해당 task-main 디렉터리 기준으로 존재하지 않는다. 실제 tracked 자산은 `task-main/maps/level5/{img,pure_text,table}/`에 있다. 원본 스크립트를 그대로 실행하면 올바른 대조가 된다고 가정하지 않는다. 지도 추출/대응은 P2 데이터 준비에서 별도로 검증한다.

추가 읽기 전용 확인에서 level3–level8 각각 img/table/pure_text 100개씩의 basename 집합이 일치했다([목록 증거](../../logs/evaluation-p0-20260910-01/vsp-source-data-inventory.json)). level5/0의 table은 `_/#/@/*`, pure_text는 시작점·구멍·목표 좌표를 제공한다. 이 한 쌍은 시작 (4,3), 목표 (5,1), 구멍 (2,2)/(3,5)의 1-based 좌표에서 일치한다. 이는 P2에서 `level/id`에 근거한 안정적 ID와 지도 metadata를 복구할 후보 경로다. 전체 내용/이미지 대응이나 논문 split은 검증하지 않았다. 결과 파일명 검색에서는 기존 VSP 명시 경로가 없었으며, 모든 결과 파일의 내부 내용을 전수 조사한 것은 아니다. 기존 prediction과의 완전한 join은 입증하지 못했다.

## 기존 공통 코드와 재사용 경계

- `src/evaluation/`와 `tests/evaluation/`는 이미 있다. `__init__.py`는 docstring만 있고, `scoring.py`는 표준 라이브러리만 import한다. `runtime.py`와 `continuous_latent.py`는 기존 추론용 계층이다.
- 기존 `scoring.py`는 `A/true/yes → Yes`, `B/false/no → No` 정규화와 exact-match를 사용한다. CoMT 선택지 매핑, 엄격한 VSP 문법, EMMA 기호 동등성을 그대로 대체하지 못한다.
- `EvaluationScoreSummary`의 `coverage_accuracy` / `valid_gold_accuracy`는 기존 계약이다. 새 manifest 분모·incomplete 계약을 여기에 덮어씌우지 않는다.
- 두 `run_one_example()`는 이미 생성 suffix를 분리하고 `prompt_identity`, `prompt_token_length`, `generated_suffix_token_ids` 등을 반환한다. **VSP main 분기의 저장 레코드는 이 필드들을 모두 보존하지 않는다.** P2는 반환값 구현을 중복하기보다 VSP 저장/bridge 연결을 점검해야 한다. 정적 근거이며 GPU 생성 동등성은 미검증이다.
- import 경로는 `evaluation.*`이다. 현재 `pytest.ini`에는 pythonpath 설정이 없다. `PYTHONPATH=src` 없이 실행한 첫 scorer 테스트는 collection 단계에서 실패(exit 2). [실패 로그](../../logs/evaluation-p0-20260910-01/existing-scoring-tests.log)를 보존했다. 소스를 고치지 않고 `PYTHONPATH=src`로 import witness와 동일 테스트를 실행해 exit 0 / 11 passed를 확인했다.

재현 명령:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_scoring.py -q -p no:cacheprovider
```

## 후속 벤치마크의 근거와 미확정 사항

| 벤치마크 | 실제 코드 경로 / 관찰 | 후속 단계 조건 |
|---|---|---|
| CoMT | `evaluate.py::run`이 원본 `id/option/answer/type`을 읽고 `judge_answer` 호출. 범주는 creation/deletion/update/selection. 정수 answer를 A–D로 변환. 빈 범주 나눗셈 방어 없음 | 원본 metadata join과 정답 타입 확인. 라이선스 미확인 소스 복사 보류. 현재 보고는 정적 분석 |
| EMMA | `evaluation/evaluate.py::gen_true_false` → `fast_extract_answer` → `evaluation/utils.py::is_equal`, 또는 별도 GPT 정오답 judge. marker 뒤 `.split('.')[0]`이 존재. 동등성은 소문자/단어 숫자 변환, 수치 소수 둘째자리 반올림, symbolic simplify 경로를 포함 | `0.5` 소수점 사례를 실제 테스트로 재현해야 함. full/mini 명시 선택, judge 별도 설정, 라이선스 확인. `utils.py`는 repo 루트가 아니라 `evaluation/` 아래 |
| VisuLogic-Eval | `ModelEvaluator.evaluate` → box/lang/GPT extraction → label 비교. constructor에서 선언되지 않은 `judege_api_key`/`judge_base_url` 참조, `get_model_name()` 정의 없음, GPT 무한 재시도 경로가 있음 | 원본 wrapper 통째 이식 금지. rules-only와 LLM extraction 분리. wrapper 전체 runtime은 실행하지 않음 |
| VLMEvalKit | `image_mcq.py::VisuLogic.evaluate`는 `utils/visulogic.py::VisuLogic_acc`로 집계. judge를 생성할 수 있어도 이 집계 함수에는 전달하지 않음. 해당 함수는 rules 추출을 직접 호출하며 0분모 방어 없음 | 모델 옵션 이름만으로 LLM 채점이라고 주장하지 않기. Position→Positional, Quantity→Quantitative, Style→Stylistic Reasoning 명시 mapping 필요 |
| Zebra 2D | Bagel의 tracked 67파일 전체 목록과 `jigsaw|visual[ _-]*search` 텍스트 검색에서 전용 scorer를 찾지 못함. `EVAL.md`와 shell wrapper는 있으나 참조하는 루트 `eval/`는 없음 | 이 checkout 범위에서의 부재이며 모든 공개 자료의 부재로 일반화하지 않음. 스키마 확보와 실제 scorer 구현을 구분 |

Zebra 검색 기록: [bagel-scorer-search.log](../../logs/evaluation-p0-20260910-01/bagel-scorer-search.log). README/원본 코드에 적힌 설치·실행 안내는 ILVR 작업 권한으로 채택하지 않았다. 외부 유료 judge를 호출하지 않았다.

## 데이터 소재

[data-inventory.json](../../logs/evaluation-p0-20260910-01/data-inventory.json)에서 현재 `data/TRAIN.jsonl` 3,450행과 `data/TEST.jsonl` 400행의 top-level 필드를 전수 확인했다. 모두 `text_input/image_input/sequence_plan/original_final_answer`만 있으며 `id/source_id/option/options/type/map_desc`는 없다. 따라서 현재 JSONL을 곧바로 VSP manifest 또는 원본 CoMT 4범주 manifest로 취급할 수 없다. `data/comt.tar.gz`와 `data/images_comt/`는 존재한다. 아카이브 전체 재해제나 원본 ID 재구성은 수행하지 않았다.

Zebra 스키마는 사용자 경로 수정에 따라 [data/evaluation/zebra_2d_source_schema.json](../../data/evaluation/zebra_2d_source_schema.json)에 있다. dataset revision은 `f6d1defc169d19180a69f517925ed6cbb06a1c97`, Jigsaw/Search 예시 각 3개다. 예시 revision, 논문 평가 split, OOD 여부는 미확인이다. EMMA/VisuLogic/VSP의 평가용 준비 manifest는 이번 조사에서 확인하지 못했다. 다른 벤치마크의 데이터 미확보는 P1 합성 fixture 기반 CPU 작업을 막지 않는다.

상위 문서가 언급하는 `migration_plan_ko.md`와 과거 `probe_results.json`은 현재 `documents/`, `docs/`에서 발견하지 못했다. 과거 통과 개수를 현재 검증에 합산하지 않았다.

## 상위 명세에 적용할 delta

1. 데이터·이미지·manifest·dataset cache는 `data/`; 응답·점수·표는 `results/evaluation/<run-id>/`; 실행 로그는 `logs/evaluation/<run-id>/`. `outputs/evaluation/` 예시는 사용하지 않는다.
2. P1은 새 독립 오프라인 모듈을 기존 `evaluation` 패키지에 추가하되 기존 scorer/추론 모듈/CLI를 보존한다. suffix 반환은 이미 있어 P2 재사용 대상으로 바꾼다.
3. 모든 Python 실행은 지정 venv를 사용한다. 원본 환경 파일 전체 설치나 별도 Python 환경 생성은 현재 승인 범위가 아니다. Gym 대조는 blocked로 남기며 dependency 변경을 암묵적으로 수행하지 않는다.
4. 신규 CPU 테스트는 경량 새 모듈만 import한다. 기존 `tests/evaluation/`의 latent 테스트까지 전부 torch-free라고 주장하지 않는다. 명시적 테스트 경로와 `PYTHONPATH=src`를 사용한다.
5. Sample 내부에 자기 자신의 manifest hash를 넣지 않는다. 파일의 완성된 bytes에 대한 manifest hash를 Prediction/run metadata에 두어 순환 hash 계약을 피한다.
6. 라이선스 미확인과 각 benchmark 원본 데이터/프로토콜의 미확정 사항은 해당 후속 단계의 조건으로 남긴다.

## 검증 상태

| 항목 | 상태 |
|---|---|
| reference root/origin/SHA/clean/license/evidence inventory | 확인 완료; lock hash 재검증은 최종 로그 참조 |
| 기존 scorer CPU 테스트 | pass: 11개, exit 0; 최초 import 실패와 구분 |
| 현재 VSP 함수 조각 진단 | 14개 관찰 저장; 전체 CLI 검증 아님 |
| 신규 evaluator CPU 테스트 | not_run: 구현 전 |
| upstream_environment | blocked: Gym 미설치, 원본 환경/실행 경로 차이 |
| gpu_smoke / full_benchmark | not_run |
| paper_protocol_verified | false |

P0 다음 결정은 [design commitment v2](../superpowers/specs/2026-09-10-ilvr-eval-foundation-vsp-design.md)의 승인이다. [비평 기록](p1-design-review.md)에 Gym first-nail 미실행, 실데이터 준비와 scorer 착수 순서의 미해결 이견, cold read에 따른 수정 사항을 남겼다. 상세 implementation plan은 그 승인 뒤 작성한다.

문서/lock을 현재 작업 루트에서 준비한 단계이므로 development branch finishing은 적용하지 않는다. 최종 확인은 reference hash/root/origin/clean 재검증, 문서 링크/동결 hash 검사, 기존 구현 소스 hash 불변 확인이다. 이미 실행한 동일 scorer 테스트를 연속 반복하지 않는다. 최종 결과는 `logs/evaluation-p0-20260910-01/final-verification.log`에 남긴다.
