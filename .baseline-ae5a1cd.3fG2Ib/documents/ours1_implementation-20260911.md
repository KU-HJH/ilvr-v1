# ours1의 Interleaved Vision-Language Reasoning과 V-off 구현

작성일: 2026-09-11
기준 구현: `ours1` commit `b580a393a5860ca74f9bd3f2944036cf59cb2e49`

이 문서의 목적은 ILVR 전체 흐름 안에서 현재 V-off 구현을 이해하고 감사할 수
있게 설명하는 것이다. 완전한 학습 재현 runbook이나 성능 보고서는 아니며,
실행 중인 실험 설정은 구현이 실제로 어떻게 켜졌는지 보여 주는 snapshot으로만
기록한다.

## 1. 먼저 읽을 핵심 요약

`ours1`은 원본 이미지와 텍스트만 한 번 처리하는 일반적인 VLM 추론과 달리,
텍스트 reasoning 사이에 **연속 벡터 형태의 시각 latent**를 반복해서 끼워 넣는다.
따라서 모델의 reasoning은 다음처럼 진행된다.

```text
원본 이미지 + 질문
        ↓
텍스트 0 → 시각 latent 1 → 텍스트 1 → 시각 latent 2 → 텍스트 2 → ...
```

각 시각 latent는 별도의 이미지 파일을 다시 읽은 결과가 아니다. Stage 2에서는
모델이 지금까지의 원본 이미지, 질문, 텍스트, 이전 latent를 보고 다음 hidden
state를 생성하며, 그 hidden state를 다음 입력 embedding으로 되먹임한다. 이것이
현재 구현에서 말하는 **Interleaved Latent Visual Reasoning(ILVR)** 이다.

현재의 V-off는 이 전체 추론에서 모든 시각 정보를 제거하지 않는다. 특정
reasoning step의 정답 텍스트 바로 앞에 있는 **현재 latent block 하나**만
국소적으로 끈다. 원본 이미지, 이전 텍스트, 이전 step의 latent는 그대로 남는다.

새로 추가한 boundary-off 모드에서는 V-off branch에서 다음 세 가지를 함께
적용한다.

1. `<|latent_start|>`를 root attention 대상에서 제외한다.
2. 현재 block의 모든 continuous latent state를 root attention 대상에서 제외한다.
3. `<|latent_end|>`의 token embedding을 사용하지 않고, 같은 위치의 replay query를
   영벡터로 시작한다.

즉, V-off 점수에는 현재 latent의 내용과 두 경계 token의 정체성이 들어가지
않는다. 다만 target token의 위치를 clean branch와 정확히 맞추기 위해 원래
sequence 길이와 mRoPE 위치는 유지한다. 따라서 이 구현은 latent block을
sequence에서 물리적으로 삭제하는 방식이 아니다.

### 이 문서에서 쓰는 구현 용어

| 용어 | 뜻 |
| --- | --- |
| clean | latent를 포함한 원래 interleaved sequence와 정상 causal attention |
| root | continuous latent rollout까지 끝난 clean sequence의 전체 decoder 입력 |
| root KV | clean final decoder에서 얻어 sidecar가 공유하는 key/value cache |
| sidecar | root KV를 공유하면서 target suffix만 다시 계산하는 context-off branch |
| replay query | 같은 정답을 teacher forcing으로 채점하기 위해 sidecar에 다시 넣는 suffix row |
| logical position | 물리적으로 sidecar 뒤에 붙어 있어도 clean sequence에서 유지하는 원래 mRoPE 위치 |

## 2. ILVR이 해결하려는 문제

여러 단계의 시각 reasoning에서는 중간 상태가 계속 변한다. 매 단계마다 새로운
중간 이미지를 vision encoder로 다시 읽으면 표현은 풍부하지만 계산량이 크다.
반대로 이미지 전체를 한 번에 지나치게 압축하면 단계별로 필요한 세부 단서를
잃기 쉽다.

이 저장소의 ILVR 경로는 두 아이디어를 결합한다.

- Stage 1은 ground-truth helper image에서 현재 reasoning에 관련된 시각 feature를
  골라 latent target을 만든다.
- Stage 2는 helper image 없이 모델 자체가 continuous latent를 autoregressive하게
  생성하고, 이를 텍스트 사이에 끼워 넣어 다음 reasoning에 사용한다.

데이터의 `sequence_plan`이 아래와 같다고 하자.

```text
text → helper_image → text → helper_image → text
```

모델에 들어가는 assistant sequence는 helper image 자리를 고정 길이 latent
scaffold로 바꾼다.

```text
T_0
<|latent_start|> <|latent_pad|> ... <|latent_pad|> <|latent_end|>
T_1
<|latent_start|> <|latent_pad|> ... <|latent_pad|> <|latent_end|>
T_2
```

현재 학습 설정의 latent size가 $K=8$이면 각 block에는 `<|latent_pad|>` 자리가
정확히 8개 있다. 이 token들은 Stage 2 실행 중 실제 continuous hidden state로
교체된다.

## 3. 전체 학습 구조

### 3.1 Stage 1: helper image로 latent 표현을 가르치기

Stage 1에서 helper image는 정답을 직접 생성하는 입력이라기보다, 좋은 latent가
어떤 시각 정보를 담아야 하는지 정하는 teacher source로 사용된다. EMA teacher는
원본 이미지와 텍스트 context를 읽고, helper image의 visual feature 후보 중 현재
reasoning과 관련된 후보를 선택한다. Dataset의 각 `helper_image` entry는
`sequence_plan`의 대응 latent block에 연결된다.

reasoning step $s$의 query를 $q_s$, helper-image feature 후보를
$f_{s,1},\ldots,f_{s,M}$이라 하자. 구현은 cosine similarity와 softmax로 후보
확률을 만든다.

$$
a_{s,j}
=
\frac{\langle f_{s,j},q_s\rangle}
{\lVert f_{s,j}\rVert\,\lVert q_s\rVert},
\qquad
\pi_{s,j}
=
\frac{\exp(a_{s,j})}
{\sum_{k=1}^{M}\exp(a_{s,k})}.
$$

$q_s$는 사용자 질문과 원본 이미지에서 얻은 global intent에 현재/이전 textual
history와 이전에 선택한 latent history를 평균해 만든다. 후보를 확률이 큰
순서로 정렬한 뒤, 누적 확률이 coverage $p$ 이상이 되는 최소 개수를 선택한다.
선택 개수는 latent block의 최대 크기 $K$를 넘지 않는다.

$$
k_s
=
\min\left(
K,
\min\left\{n:\sum_{j=1}^{n}\pi_{s,(j)}\ge p\right\}
\right).
$$

선택한 EMA-teacher feature는 student의 latent 입력 위치에 teacher-forcing target으로
주입된다. Student가 바로 앞 위치에서 예측한 hidden state와 target latent의
방향을 cosine loss로 맞춘다.

$$
\mathcal L_{\mathrm{align}}
=
1-
\frac{1}{N}
\sum_{i=1}^{N}
\cos\left(\hat z_i, z_i^{\mathrm{teacher}}+\epsilon_i\right),
\qquad
\epsilon_i\sim 0.01\,\mathcal N(0,I).
$$

$k_s<K$이면 선택 feature는 block의 앞쪽 $k_s$개 latent 위치에만 순서대로
들어간다. 이 위치만 latent alignment mask가 `true`이고, 나머지 $K-k_s$개
`<|latent_pad|>` 위치에는 별도의 alignment target을 주지 않는다. Stage 1 CE에서도
`<|latent_pad|>` token은 mask되므로, 선택되지 않은 자리를 pad token으로 맞히도록
학습하지 않는다.

Stage 1의 전체 loss는 다음 형태다.

$$
\mathcal L_{\mathrm{Stage1}}
=
\lambda_{\mathrm{CE}}\mathcal L_{\mathrm{CE}}
+
\lambda_{\mathrm{sim}}\mathcal L_{\mathrm{align}}.
$$

EMA teacher는 성공한 optimizer step 뒤에 설정된 decay로 갱신된다. Decay를 1로
설정하면 teacher가 고정되고, 1보다 작게 설정하면 student update를 따라 이동한다.

### 3.2 Stage 2: helper image 없이 continuous latent 생성하기

Stage 2 데이터에는 원본 사용자 이미지만 들어간다. Ground-truth helper image는
Stage 2 processor 입력에서 제거되며, helper-image tensor가 유입되면 trainer가
오류를 낸다.

모델은 먼저 원본 이미지 token을 frozen vision tower의 feature로 한 번 교체한다.
그 다음 각 latent 위치를 왼쪽부터 순서대로 생성한다. step $s$의 $j$번째 latent
위치를 $p_{s,j}$라고 하면 개념적으로 다음 recurrence를 수행한다.

$$
z_{s,j}
=
h_\theta\!\left(
C_{<p_{s,j}};
z_{<s,*},z_{s,<j}
\right)_{\mathrm{last}}.
$$

여기서 $C_{<p_{s,j}}$는 해당 위치 전까지의 원본 이미지, 질문, 텍스트, control
token을 포함한 causal prefix다. 이미 생성한 latent state는 다음 호출의 입력
embedding에 다시 삽입된다. 현재 구현은 KV cache를 사용하지 않고 이 과정을
미분 가능하게 반복한다.

모든 latent 위치가 채워지면 완성된 embedding sequence를 final decoder에 넣어
clean logits와 CE를 계산한다. `SUPERVISE_LATENT_BOUNDARIES=true`이면
`<|latent_start|>`와 `<|latent_end|>`도 clean CE target으로 학습하고,
`<|latent_pad|>` 위치는 CE에서 제외한다.

### 3.3 실제 추론에서 text와 latent가 교차하는 방법

학습 데이터에는 미리 만든 latent scaffold가 있지만, 추론에서는 모델이 먼저
일반 text token을 한 개씩 생성한다. 이때 `<|latent_pad|>`와 `<|latent_end|>`는
직접 sampling하지 못하게 막고, `<|latent_start|>`는 모델이 선택할 수 있다.

모델이 `<|latent_start|>`를 생성하면 controller가 다음 절차를 수행한다.

```text
모델이 <latent_start> 생성
        ↓
controller가 K개의 <latent_pad>와 <latent_end> scaffold를 자동으로 추가
        ↓
K개 pad 위치를 continuous latent z_1, ..., z_K로 순차 교체
        ↓
<latent_end> 뒤에서 다시 일반 text token 생성
```

$K$는 checkpoint config의 `latent_size`이며 현재 실험에서는 8이다. 모델이 다음
`<|latent_start|>`를 생성하면 같은 과정이 반복되므로, text와 latent block이
동적으로 여러 번 interleave될 수 있다. EOS, `max_new_tokens`, 또는 최대 sequence
길이에 도달하면 추론을 끝낸다.

따라서 control token의 역할은 분명히 나뉜다.

- `<|latent_start|>`: 모델이 “지금 latent reasoning을 시작하겠다”고 선택하는 token
- `<|latent_pad|>`: controller가 확보한 뒤 continuous hidden state로 교체하는 자리
- `<|latent_end|>`: 고정 길이 latent block이 끝났고 text 생성으로 돌아갈 위치

### 3.4 Stage 2A와 Stage 2B

- Stage 2A 또는 ordinary Stage 2 경로는 clean sequence의 CE를 학습한다.
- Stage 2B `context_preference` 경로는 같은 정답 텍스트를 clean context와
  context-off branch에서 각각 채점해, clean context가 더 유용하도록 학습한다.
- 현재 boundary-off 실험은 `STAGE2B_DIRECT_FROM_STAGE1=true`이므로 Stage 2A
  checkpoint를 거치지 않고 Stage 1 adapter에서 직접 Stage 2B를 시작했다.

## 4. Stage 2B의 clean, V-off, T-off

Reasoning step $s$를 다음과 같이 표현하자.

```text
... 이전 context ... | T_prev | <latent_start> z_1 ... z_K <latent_end> | Y_s |
```

$Y_s=(y_{s,1},\ldots,y_{s,N_s})$는 이 latent block 바로 뒤의 정답 textual step이다.
세 branch는 모두 **동일한 $Y_s$**를 teacher forcing으로 채점한다.

| Branch | $Y_s$를 채점할 때 제거하는 정보 |
| --- | --- |
| clean | 없음 |
| V-off | 현재 step의 latent block |
| T-off | 현재 latent 직전의 textual block |

여기서 V-off는 original image 전체를 끄지 않는다. 또한 이전 reasoning step의
latent도 유지한다. 현재 step의 latent가 바로 다음 텍스트에 주는 국소적인 기여를
비교하기 위한 counterfactual branch다.

일반 context-preference는 V-off와 T-off를 모두 만든다. 반면
`STEP_VISUAL_ONLY_CONTEXT_PREFERENCE=true`인 V-only 모드는 T-off group을 아예
생성하지 않으며 clean과 V-off만 계산한다.

## 5. 왜 기존 V-off만으로는 경계 token이 남았는가

한 latent block의 token 위치를 다음처럼 정의한다.

```text
위치:       s        s+1             e-1       e        e+1 ...
token:  latent_start  latent states z_1 ... z_K  latent_end  y_1 ...
```

`ReasoningStepSpan`의 좌표와 대응시키면 다음과 같다.

$$
s=\texttt{latent\_start}-1,
\qquad
e=\texttt{latent\_end},
\qquad
\texttt{target\_text\_start}=e+1.
$$

이름이 비슷해 생길 수 있는 혼동을 표로 정리하면 다음과 같다. 모든 span은 Python
slice와 같은 half-open 구간이다.

| 의미 | 실제 위치/구간 |
| --- | --- |
| `<|latent_start|>` marker | $s=\texttt{span.latent\_start}-1$ |
| continuous latent positions | $[\texttt{span.latent\_start},\texttt{span.latent\_end})=[s+1,e)$ |
| `<|latent_end|>` marker | $e=\texttt{span.latent\_end}$ |
| target text positions | $[\texttt{span.target\_text\_start},\texttt{span.target\_text\_end})$ |

즉, `span.latent_start`는 start marker의 위치가 아니라 **첫 latent state의 위치**이고,
`span.latent_end`는 latent-state slice의 exclusive end이면서 동시에 end marker의
위치다.

기존 V-off는 root KV에서 구간

$$
[\texttt{latent\_start},\texttt{latent\_end})=[s+1,e)
$$

만 attention 차단했다. 따라서 continuous latent state $z_1,\ldots,z_K$는 보이지
않았지만 다음 두 content path가 남았다.

- `<|latent_start|>`의 root KV는 차단 구간 밖이므로 보였다.
- 첫 target $y_{s,1}$을 예측하는 sidecar query는 clean sequence의
  `<|latent_end|>` embedding을 그대로 replay했다.

그러므로 기존 방식은 정확히 말해 **latent-state-off**였고, latent block의 두
control token까지 완전히 끈 방식은 아니었다.

## 6. 현재 boundary-suppressed V-off의 정확한 동작

### 6.1 Root attention 차단 범위 확대

새 모드는 sidecar metadata의 차단 구간을 다음처럼 넓힌다.

$$
B_s^{\mathrm{new}}
=
[s,e+1)
=
[\texttt{latent\_start}-1,\texttt{latent\_end}+1).
$$

따라서 metadata 수준에서는 `<|latent_start|>`, 모든 latent state,
`<|latent_end|>`를 포함한 전체 block을 차단한다.

각 sidecar query가 볼 수 있는 root key 집합은 다음과 같다.

$$
\mathcal A_s^{\mathrm{root}}
=
\left\{k\mid 0\le k<e\right\}
\setminus B_s^{\mathrm{new}}.
$$

실제 root ancestor의 끝이 $e$이므로 root cache의 `<|latent_end|>` 위치 자체는
원래도 보이지 않는다. 중요한 차이는 `<|latent_start|>`가 새로 root에서
제외되고, 아래 설명처럼 replay된 `<|latent_end|>` embedding도 제거된다는 점이다.

Sidecar 내부의 replay query는 자기 branch 안에서만 causal attention을 한다.
$r$번째 query까지의 local row를 $q_0,\ldots,q_r$라고 하면 전체 허용 집합은
다음과 같다.

$$
\mathcal A_{s,r}
=
\mathcal A_s^{\mathrm{root}}
\cup
\{q_0,\ldots,q_r\}.
$$

Dense backend는 이 식을 boolean visibility tensor로 만들고, Flex backend는 같은
조건을 `BlockMask` 함수로 만든다.

한 step의 clean root와 V-off sidecar 관계를 물리적 배치로 보면 다음과 같다.

```text
clean root KV: [원본 이미지/질문 ... T_prev | start z_1 ... z_K end | Y_s ...]
                                                     ×    ×       ×
V-off가 쓰는 root: [원본 이미지/질문 ... T_prev]       현재 block은 보지 않음
sidecar suffix:                                         [q_0, q_1, ..., q_(N-1)]
                                                        0    y_1       y_(N-1)
예측 target:                                            y_1  y_2       y_N
```

Clean root KV 자체를 다시 만들지는 않지만, attention mask가 `×`로 표시한 현재
block의 key/value를 V-off query가 읽지 못하게 한다.

### 6.2 `<|latent_end|>` replay embedding 영벡터화

Causal LM에서 첫 target token을 예측하려면 target 직전 위치에 query row가 하나
필요하다. Clean branch에서는 그 위치가 `<|latent_end|>`다. 이 row를 통째로
삭제하면 이후 target의 causal 위치가 clean branch와 달라진다.

따라서 새 V-off는 row와 logical position은 유지하되 첫 입력 embedding만
영벡터로 바꾼다.

$$
x_{q_r}^{\mathrm{Voff}}
=
\begin{cases}
\mathbf 0, & r=0,\\
E(y_{s,r}), & 1\le r<N_s.
\end{cases}
$$

$q_0$은 logical position $e$에서 $y_{s,1}$을 예측한다. $r\ge1$인 row는 앞선
정답 token을 teacher forcing 입력으로 사용해 다음 token을 예측한다.

| Replay row | 입력 embedding | 예측 target |
| --- | --- | --- |
| $q_0$ | $\mathbf 0$ | $y_{s,1}$ |
| $q_1$ | $E(y_{s,1})$ | $y_{s,2}$ |
| $\cdots$ | $\cdots$ | $\cdots$ |
| $q_{N_s-1}$ | $E(y_{s,N_s-1})$ | $y_{s,N_s}$ |

이 때문에 `<|latent_end|>`의 token ID나 embedding은 V-off score에 들어가지 않는다.
다만 $q_0$ 자체는 영벡터에서 시작한 뒤 허용된 이전 context에 attention하므로,
첫 target은 원본 이미지와 이전 reasoning context를 사용해 예측할 수 있다.

### 6.3 위치와 sequence geometry 보존

새 모드는 embedding content만 개입한다. Sidecar replay는 clean branch의 원래
logical mRoPE position을 그대로 가져온다.

$$
\operatorname{pos}(q_r)=e+r.
$$

따라서 보존되는 것은 다음과 같다.

- target token의 absolute/logical position
- latent block이 차지하던 길이
- target 내부의 causal 순서
- clean과 V-off 사이의 동일한 scoring alignment

반대로 제거되는 것은 현재 latent state와 두 control token의 **내용 경로**다.
그러므로 “`<|latent_end|>`의 영향이 없다”는 말은 token identity와 embedding
content가 V-off prediction에 영향을 주지 않는다는 뜻이다. `<|latent_end|>`가 있던
논리적 위치와 한 개의 neutral query row까지 삭제한다는 뜻은 아니다.

### 6.4 정보 누출 방지 조건

Target text 사이에 다른 clean token이 끼어 있으면 그 token의 hidden state가
latent 정보를 중계할 수 있다. 이를 막기 위해 boundary suppression은 반드시
다음을 요구한다.

$$
\texttt{target\_text\_start}=\texttt{latent\_end}+1.
$$

Runtime validation은 추가로 다음을 확인한다.

- branch가 `context_preference`의 `visual_off`인지
- T-off group이 비어 있는지
- 모든 V-off branch가 동일하게 boundary suppression을 사용하는지
- start/end token ID와 latent-pad 개수가 canonical block과 일치하는지
- replay token, source position, logical position, target position이 원본 clean
  sequence의 teacher-forcing 구조와 정확히 일치하는지
- 서로 다른 reasoning step의 start/end marker를 잘못 묶지 않았는지

## 7. 점수와 학습 loss

### 7.1 Step score

Clean과 V-off는 같은 target $Y_s$의 token 평균 log-probability를 사용한다.

$$
c_s
=
\frac{1}{N_s}
\sum_{r=1}^{N_s}
\log p_\theta\!\left(y_{s,r}\mid C_s^{\mathrm{clean}},y_{s,<r}\right),
$$

$$
v_s
=
\frac{1}{N_s}
\sum_{r=1}^{N_s}
\log p_\theta\!\left(y_{s,r}\mid C_s^{\mathrm{Voff}},y_{s,<r}\right).
$$

Token 수가 긴 step이 더 큰 가중치를 받지 않도록 먼저 step 안에서 평균하고,
그 다음 reasoning step들을 동일 가중치로 평균한다.

### 7.2 Context-preference loss

V-off와 비교한 clean advantage는

$$
\Delta_s^{V}=c_s-v_s
$$

이고, 구현된 preference loss는 다음과 같다.

$$
\mathcal L_V
=
\frac{1}{S}
\sum_{s=1}^{S}
-\log\sigma\!\left(\beta(\Delta_s^V-m)\right).
$$

$\beta$는 preference의 민감도, $m$은 원하는 clean–V-off margin이다. 이 loss는
reference model이나 generated rejected answer를 사용하지 않는다. $c_s$와 $v_s$
모두 같은 학습 모델에서 계산되며 어느 쪽도 detach하지 않는다.

따라서 이 목적함수는 $c_s$를 높이는 방향과 $v_s$를 낮추는 방향 모두로 줄어들 수
있다. “latent를 활용하게 한다”는 표현은 clean–V-off 간격을 키우는 **최적화
의도**를 뜻한다. 실제로 유용한 latent reasoning이 향상됐는지는 별도의 held-out
평가로 확인해야 한다.

### 7.3 선택적인 entropy loss

각 target 위치의 분포 entropy를

$$
H(p)=-\sum_{w\in\mathcal V}p(w)\log p(w)
$$

라고 하자. V-off가 clean보다 설정된 entropy margin $\eta$만큼 더 불확실해지도록
하는 항은 step별로 다음과 같다.

$$
\mathcal L_{H,V}
=
\frac{1}{S}
\sum_{s=1}^{S}
\frac{1}{N_s}
\sum_{r=1}^{N_s}
\max\left(0,\eta-H_{s,r}^{\mathrm{Voff}}+H_{s,r}^{\mathrm{clean}}\right).
$$

### 7.4 V-only 전체 objective

V-only 모드에서는 T-off 항이 없으므로 전체 loss는 다음과 같다.

$$
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{CE}}
+
\lambda_P\mathcal L_V
+
\lambda_H\mathcal L_{H,V}.
$$

2026-09-11에 시작한 현재 boundary-off 학습은
$\beta=0.1$, $m=0$, $\lambda_P=1$, $\lambda_H=0$이므로 실제 최적화 식은
다음처럼 단순해진다.

$$
\boxed{
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{CE}}
+
\frac{1}{S}
\sum_{s=1}^{S}
-\log\sigma\!\left(0.1(c_s-v_s)\right)
}
$$

## 8. Gradient를 해석할 때 주의할 점

Boundary suppression이 보장하는 범위는 **V-off sidecar score $v_s$의 현재 latent
block 경로**다. $v_s$만 backward했을 때 현재 block의 continuous latent state,
`<|latent_start|>` embedding, `<|latent_end|>` embedding으로 가는 gradient는 0이다.

$$
\frac{\partial v_s}{\partial z_{s,j}}=0,
\qquad
\frac{\partial v_s}{\partial E(\texttt{latent\_start})}=0,
\qquad
\frac{\partial v_s}{\partial E(\texttt{latent\_end})}=0.
$$

그러나 전체 학습 loss에서 이 parameter들의 gradient가 항상 0이라는 뜻은 아니다.
Clean CE와 clean score $c_s$는 원래 latent block을 그대로 사용한다. 따라서
다음 경로는 의도적으로 살아 있다.

```text
clean CE / clean score
        ↓
latent state와 boundary token을 정상적으로 학습
```

즉, 이 모드는 clean reasoning 자체의 latent를 없애는 것이 아니라, clean과 비교할
counterfactual V-off branch에서만 현재 latent와 boundary content를 제거한다.

Clean CE와 clean score의 관계도 구분할 필요가 있다.

- Clean CE는 assistant content 전체에서 label이 살아 있는 text와, 설정에 따라
  boundary token을 감독한다. 현재 설정에서는 모든 `<|latent_pad|>`만 CE에서
  제외하고 두 boundary marker는 포함한다.
- Clean score $c_s$는 preference 비교 대상인 각 $Y_s$의 text token만 다시
  선택해 평균한다.
- 따라서 $Y_s$의 text token은 CE 경로와 preference의 clean-score 경로에 모두
  기여하지만, 두 항의 reduction과 목적은 서로 다르다.

## 9. Shared-prefix 실행 흐름

각 rank-local microbatch는 현재 반드시 batch size 1이다. 한 sample에서 다음
순서로 계산한다.

```text
1. 원본 이미지를 frozen vision tower로 한 번 encoding
2. continuous latent를 순차 rollout
3. 완성된 clean sequence를 final decoder에 통과
4. clean target logits, CE, clean prefix KV 확보
5. 같은 KV를 공유하는 V-off suffix sidecar 실행
6. clean/V-off score로 preference loss 계산
```

V-only에서는 T-off sidecar가 없으므로 logical scoring group은 clean 1개와 V-off
1개다. Sidecar는 target suffix만 replay하며 전체 clean sequence를 별도 sample로
복제하지 않는다. Dense와 Flex backend는 같은 visibility 의미를 구현한다.

현재 context-preference 경로는 다음을 수행하지 않는다.

- reference model forward
- negative answer 생성
- rejected completion scoring
- preference rollout

## 10. 설정이 runtime까지 전달되는 경로

아래는 boundary suppression을 켜는 데 필요한 **최소 모드 조합**이다. 완전한
재현 명령이 아니라 기존 Stage 2B launch에 추가하는 예시다.

```bash
STEP_PREFERENCE_ENABLED=true \
STEP_PREFERENCE_MODE=stage2b \
STEP_PREFERENCE_OBJECTIVE=context_preference \
STEP_VISUAL_ONLY_CONTEXT_PREFERENCE=true \
STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES=true \
bash run_stage2_training.sh
```

Stage 1에서 직접 시작하려면 여기에 `STAGE2B_DIRECT_FROM_STAGE1=true`, base model,
Stage 1 adapter, 데이터, output, optimizer/분산 설정을 함께 지정해야 한다.

전달 경로는 다음과 같다.

```text
environment variable
  → run_stage2_training.sh의 boolean 검증
  → CLI flag
  → StepPreferenceConfig
  → Stage2Trainer._compute_context_preference_loss
  → build_context_sidecar_plan
  → Qwen shared-prefix runtime validation
  → root attention mask + zero first replay embedding
  → V-off score와 loss
```

`STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES=true`는 다음 조건 밖에서는 즉시
거부된다.

- step preference 활성화
- Stage 2B
- `context_preference` objective
- V-only context preference 활성화

Multi-node launcher도 해당 environment variable을 allowlist에 포함해 양쪽 node의
worker command로 전달한다.

## 11. 현재 실행 중인 실험의 구현 관련 설정

Run ID:
`ours1-ctx-pref-vonly-boundaryoff-direct-s1-dora-egb16-lr5e5-s212-20260911t085315z`

이 run의 완전한 launch snapshot은
`logs/ours1-ctx-pref-vonly-boundaryoff-direct-s1-dora-egb16-lr5e5-s212-20260911t085315z/launch.sh`
에 있다.

| 항목 | 값 |
| --- | --- |
| 초기화 | Stage 1 DoRA adapter에서 직접 시작 |
| Objective | CE + V-only context preference |
| Boundary-suppressed V-off | 활성화 |
| T-off | 생성하지 않음 |
| Entropy loss | 비활성화, $\lambda_H=0$ |
| Latent size | 8 |
| Clean latent boundary CE | 활성화 |
| Tuning | DoRA, rank 32, alpha 64, dropout 0.05 |
| Learning rate | $5\times10^{-5}$ |
| Rank-local batch | GPU당 1 |
| World size | 4 GPUs |
| Gradient accumulation | 4 |
| Effective global batch | 16 |
| Sidecar attention | FlexAttention |
| Language / vision attention | SDPA / FlashAttention 2 |

## 12. 구현 파일별 역할

| 파일 | 역할 |
| --- | --- |
| `src/step_preference/spans.py` | text–latent–text reasoning step의 token 구간 정의 |
| `src/stage1/targets.py` | EMA teacher의 helper-image feature 선택과 latent target 구성 |
| `src/stage1/trainer.py` | Stage 1 CE + latent alignment loss와 EMA lifecycle |
| `src/stage2/data.py` | helper image 제거, latent scaffold와 label/span 구성 |
| `transformers/.../stage2_latent_rollout.py` | Stage 2 continuous latent recurrence |
| `src/evaluation/continuous_latent.py` | 추론 중 text 생성과 latent block 전환 제어 |
| `src/step_preference/sidecar.py` | V-off/T-off plan과 Dense/Flex visibility 정의 |
| `transformers/.../stage2_shared_prefix.py` | prefix KV 공유, V-off replay, 첫 embedding zeroing, branch 검증 |
| `src/stage2/trainer.py` | clean/V-off scoring과 context-preference loss reduction |
| `src/step_preference/config.py` | 모드 조합의 type 및 semantic validation |
| `src/utils_deepseed.py` | public CLI flag 정의 |
| `run_stage2_training.sh` | environment parsing, 조합 검증, CLI forwarding |
| `src/multinode_runtime.py` | multi-node environment forwarding allowlist |

위 표의 `transformers/...`는
`transformers/src/transformers/models/qwen2_5_vl/`을 뜻한다.

## 13. 검증된 동작

구현에는 다음 regression test가 포함되어 있다.

- V-only plan이 V-off만 만들고 T-off를 만들지 않는지 확인
- 새 차단 범위가 두 marker와 latent states를 모두 포함하는지 확인
- `<|latent_start|>`와 `<|latent_end|>` embedding을 크게 바꿔도 V-off logits와
  entropy가 변하지 않는지 확인
- V-off score의 gradient가 현재 latent states와 두 marker embedding으로 흐르지
  않는지 확인
- target이 `<|latent_end|>`에 바로 붙지 않은 잘못된 구조를 거부하는지 확인
- 잘못된 marker interval, cross-step interval, replay metadata를 거부하는지 확인
- public config가 trainer와 실제 causal cut까지 전달되는지 확인
- Dense와 CUDA Flex backend의 score, gradient, call count가 일치하는지 확인
- multi-node worker 양쪽에 두 flag가 전달되는지 확인

구현 commit을 완료할 때 repository test suite는 `3151 passed, 10 skipped`였고,
claim-specific CUDA Dense/Flex parity test도 통과했다. 이 수치는 코드 동작에 대한
근거이며, boundary-off가 모델 품질을 향상시킨다는 실험 결과는 아니다.

이 보고서를 작성하면서 boundary plan, marker-logit 불변성, gradient 차단, config,
launcher, multi-node forwarding을 다루는 CPU focused test 7개도 다시 실행했고 모두
통과했다.

## 14. 범위와 한계

이 구현을 해석할 때 다음을 구분해야 한다.

- V-off는 **현재 step의 latent block**만 끈다. 원본 이미지와 이전 step context는
  남는다.
- Boundary suppression은 V-off sidecar에만 적용된다. Clean CE와 clean score의
  latent 및 boundary 사용은 그대로다.
- `<|latent_end|>` token content는 제거되지만 그 위치의 neutral query와 mRoPE
  geometry는 유지된다.
- Sequence에서 latent block을 삭제하지 않으므로 “block deletion ablation”과는
  다른 실험이다.
- 현재 context-preference training은 rank-local batch size 1만 지원한다.
- 테스트는 causal isolation과 수치적 backend 일치를 검증한다. 학습 효과,
  일반화 성능, 메모리 적합성은 별도의 평가 결과가 필요하다.

## 15. 결론

현재 `ours1`의 V-only boundary-off는 ILVR의 clean interleaved reasoning을 그대로
유지하면서, 각 target textual step에 대해 **현재 latent visual unit이 없었다면
어땠을지**를 같은 모델 안에서 비교하는 학습 모드다.

기존 V-off가 continuous latent states만 attention에서 가렸다면, 새 구현은
`<|latent_start|>`의 root content와 `<|latent_end|>`의 replay embedding까지
차단한다. 동시에 target의 위치와 causal geometry는 보존한다. 그 결과 V-off
점수는 현재 latent block의 내용에 의존하지 않으며, clean score와의 차이를 통해
interleaved latent visual signal의 clean–V-off 점수 간격을 키우는 최적화 압력을
준다. 이것이 실제 reasoning 품질 향상으로 이어지는지는 별도 평가의 대상이다.
