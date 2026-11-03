# Codex 프롬프트 — ILVR의 Reference/ 저장소 준비

현재 **ILVR 작업 루트**의 `Reference/`에 아래 참고 코드 저장소 7개를 준비하라. 설치된 Superpowers와 프로젝트 지침을 따르되, 이번 작업은 **저장소 준비와 검증만** 수행한다. 평가 코드 이식, 의존성 설치, 데이터셋·모델 가중치 다운로드는 하지 않는다.

## 준비할 저장소

| 로컬 경로 | 원본 URL | 신규 준비 시 버전 |
|---|---|---|
| `Reference/Visual-Spatial-Planning` | `https://github.com/UCSB-NLP-Chang/Visual-Spatial-Planning.git` | 원격 기본 브랜치 HEAD |
| `Reference/CoMT` | `https://github.com/gogoczh/CoMT.git` | 원격 기본 브랜치 HEAD |
| `Reference/EMMA` | `https://github.com/EMMA-Bench/EMMA.git` | 원격 기본 브랜치 HEAD |
| `Reference/VisuLogic-Eval` | `https://github.com/VisuLogic-Benchmark/VisuLogic-Eval.git` | 원격 기본 브랜치 HEAD |
| `Reference/VLMEvalKit` | `https://github.com/open-compass/VLMEvalKit.git` | 원격 기본 브랜치 HEAD |
| `Reference/Bagel-Zebra-CoT` | `https://github.com/multimodal-reasoning-lab/Bagel-Zebra-CoT.git` | 원격 기본 브랜치 HEAD |
| `Reference/MathRuler` | `https://github.com/hiyouga/MathRuler.git` | **`v0.1.0` 태그의 commit** |

## 수행 규칙

1. 현재 ILVR 루트와 작업 상태를 확인하고 **현재 작업 디렉터리/worktree에서** 준비한다. 새 worktree를 만들거나 ILVR의 브랜치를 바꾸지 않는다. `/Reference/`를 기존 `.gitignore`에 중복 없이 추가하되 다른 내용을 보존한다. 이미 추적 중인 Reference 파일은 자동으로 추적 해제하지 말고 충돌로 보고한다.
2. **없는 경로만 clone**한다. 신규 clone에는 `GIT_LFS_SKIP_SMUDGE=1`을 적용하고 submodule을 재귀적으로 가져오지 않는다. `references.lock.json`에 해당 저장소의 commit이 있으면 그 SHA를 사용하고, 없으면 표의 버전을 사용한다. 신규 저장소는 실제 commit SHA로 detached checkout하여 고정한다. MathRuler의 lock이 `v0.1.0`과 충돌하면 임의 선택하지 말고 해당 항목을 중단·보고한다.
3. **이미 있는 저장소는 보존**한다. Git 최상위 경로가 해당 Reference 경로와 일치하는지, origin이 표의 저장소인지, 작업 트리가 깨끗한지, 기존 lock과 일치하는지 검사한다. lock이 없으면 기존 HEAD를 유지하되 MathRuler는 `v0.1.0`과의 일치를 확인한다. 일반 폴더·잘못된 origin·로컬 수정·버전 불일치는 덮어쓰거나 `pull`, `checkout`, `reset`, `clean`으로 해결하지 말고 보고한다.
4. 참조 코드는 준비 후 읽기 전용으로 취급한다. 저장소 내 설치 스크립트나 테스트는 실행하지 않는다. 이번에는 새 lock·감사문서·구축 스크립트를 만들지 않고, **lock 생성과 코드 감사는 기존 이식 문서의 P0 단계에 남긴다.** 자동 commit·push도 하지 않는다.
5. 마지막에 7개 저장소 각각의 **경로, 실제 40자리 commit SHA, clean 여부, 신규 준비/기존 재사용/차단 상태**를 표로 보고한다. MathRuler 태그 일치와 ILVR에서 Reference가 추적 제외되는지도 검증한다. 네트워크·권한·버전 문제로 준비하지 못한 항목은 성공으로 표시하지 않는다.

저장소 준비 결과를 보고한 뒤 멈춘다. P0 감사나 평가 구현은 이 요청에서 시작하지 않는다.
