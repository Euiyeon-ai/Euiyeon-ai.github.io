+++
title = 'LSTM과 GRU — 게이트로 오래 기억하고, 더 가볍게'
date = 2026-09-15T17:00:00+09:00
draft = false
slug = 'lstm-gru-gates'
description = 'RNN은 왜 먼 과거를 배우기 어렵고, LSTM은 Cell state와 Gate 3개로 그걸 어떻게 늦췄으며, GRU는 무엇을 줄였나. 문제 → 원인 → 해결책 → 변화 순서로 읽는다.'
tags = ['부트캠프', '스터디 발표', '딥러닝', 'RNN', 'LSTM', 'GRU', 'PyTorch']
categories = ['개발']
+++

부트캠프 스터디에서 발표한 자료입니다. RNN → LSTM → GRU를 각각 **문제 → 원인 → 해결책 → 변화** 네 칸으로 읽습니다. 수식보다 "왜 그렇게 만들었나"에 집중했고, 질문은 접어 두었으니 먼저 답을 예상해 보고 펼쳐 보세요.

---

## 지금 어디에 있나

![자연어처리 모델의 발전 흐름. RNN에서 LSTM, GRU를 거쳐 Seq2Seq, Attention, Transformer로 이어진다](/images/study-lstm-gru/lstm-gru-14-roadmap.svg)

---

## 어떤 흐름으로 볼 것인가

| 묻는 것 | 뜻 |
| --- | --- |
| **문제 ← 원인** | 무엇이 문제였고, **왜** 문제였나? |
| **해결책 → 변화** | 어떻게 해결했고, 그래서 **무엇이 달라졌나**? |

![RNN, LSTM, GRU 각각의 문제, 원인, 해결책, 변화를 한눈에 정리한 표](/images/study-lstm-gru/lstm-gru-11-overview.svg)

![RNN에서 LSTM, GRU로 이어지는 계보](/images/study-lstm-gru/lstm-gru-10-lineage.svg)

---

## ① RNN

| | |
| --- | --- |
| **문제** | 일반 신경망은 **순서 · 과거**를 못 본다 |
| **원인** | 입력을 하나씩 **독립적으로** 처리 — 앞 입력의 결과가 **다음 계산에 전달되지 않는다** |
| **해결책** | 이전 `hidden state`를 **다음 시점에 전달** |
| **변화** | 과거를 기억하며 **순차 처리** 가능 |

`나 → 는 → 밥 → 을 → 먹었다`

![RNN 구조. 단어를 하나씩 읽으면서 이전 hidden state를 다음 시점으로 넘긴다](/images/study-lstm-gru/lstm-gru-01-rnn.svg)

```
h_새 = tanh( W·h_이전 + U·x_지금 )
```

| 기호 | 뜻 | 포인트 |
| :---: | --- | --- |
| **h** | 요약 메모 | **하나뿐.** 크기 고정(예: 32개), 내용은 학습 |
| **x_지금** | 이번 칸에 들어온 단어 | |
| **W · U** | 가중치 | **100칸이 전부 같은 W** (시간축 공유) |
| **tanh** | 믹서기 | 앞 메모 + 새 단어를 **통째로 갈아** 새 메모를 만든다 |

왜 하필 tanh인가? (−1 ~ 1 : 값 제한 + 정보의 방향성 전달)

- **시그모이드**는 과거 기억을 너무 빨리 잊어서 탈락
- **ReLU**는 100칸을 거치면 값이 폭발해서 탈락

---

## ② LSTM

### 1. 문제 — RNN은 먼 과거의 정보를 학습하기 어렵다

### 2. 원인 — 가중치 × 활성화함수 미분값이 시점마다 반복해서 곱해진다

**RNN** : hidden state 하나가 기억을 맡는다. 과거 정보와 기울기가 매 시점 **W → tanh** 변환을 반복해서 통과한다.

```
h_t = tanh( W·h_(t-1) + U·x_t + b )    ⇒    (W·tanh′)₁ (W·tanh′)₂ (W·tanh′)₃ …
```

- **반복 곱의 효과 < 1** → 0.5¹⁰⁰ ≈ 0 → **기울기 소실**
- **반복 곱의 효과 > 1** → 1.5¹⁰⁰ → 매우 큼 → **기울기 폭발**

→ 긴 시퀀스에서 기울기가 **소실되거나 폭발**하기 쉽다.

<details>
<summary>Q1. <b>정보가 사라지는 것</b>과 <b>기울기가 사라지는 것</b>은 어떻게 다른가?</summary>

둘은 **흐르는 방향과 역할이 다르다.**

- **정보 = 앞으로 가는 기억**
- **기울기 = 뒤로 가는 수정 신호**

</details>

### 3. 해결책 — Cell state + Gate 3개

**LSTM** : 장기 기억용 경로 **Cell state**를 따로 추가한다. 이전 기억을 매번 새로 변환하는 대신 **forget gate `f_t`를 곱해 비교적 직접 다음 시점으로** 전달한다.

```
C_t = f_t · C_(t-1) + i_t · C̃_t          C_(t-1) ──( × f_t )──▶ C_t
```

→ 중요한 기억이면 `f_t ≈ 1`로 학습 → 기울기 × 0.999 × 0.999 … → **훨씬 오래 보존.**

**구조 한눈에**

![LSTM 구조 한눈에. Cell state 통로 위에 forget, input, output 세 개의 gate가 달려 있다](/images/study-lstm-gru/lstm-gru-13-lstm-glance.svg)

<details>
<summary>Q2. Input gate를 <code>tanh × σ</code>로 나눈 이유는?</summary>

새로 기억할 **내용(무엇을)** 과 그 내용을 **얼마나 반영할지(얼마나)** 를 따로 학습하기 위해.

| **tanh ( −1 ~ 1 ) = 내용** | **σ ( 0 ~ 1 ) = 조절** |
|---|---|
| 무엇을 새로 기억할까 — 양수 · 음수를 가진 **새 정보 표현** | 얼마나 반영할까 — 내용은 맞아도 **지금 적을 때가 아닐 수** 있다 |

넣는 양 = **σ × tanh**. Gate 세 개는 전부 σ(얼마나), tanh는 "넣을 내용" 하나뿐.

</details>

<details>
<summary>Q3. forget gate 값이 1이면 "다 잊는 것"일까, "다 기억하는 것"일까?</summary>

**1 = 다 남김, 0 = 다 지움.** 이름은 '잊는 문'이지만 정하는 건 **남기는 양.**

</details>

<details>
<summary>Q4. forget gate는 왜 <b>열어둔 채로</b> 시작해야 하나?</summary>

**1. 닫혀 있으면 신호가 사라진다**

```
f = 0.5   →  0.5 → 0.25 → 0.125 → …  → 거의 0        (반쯤 닫힘 = RNN 과 같은 곱셈 반복)
f ≈ 1     →  거의 그대로 → 거의 그대로 → …  → 오래 전달
```

**2. 그래서 처음엔 열어 둔다**

forget gate는 학습하면서 조절되지만, **처음부터 너무 닫혀 있으면 과거의 신호가 사라져 "무엇을 오래 기억해야 하는지" 자체를 배우기 어렵다.** 열어야 한다는 것도 신호가 와야 배운다. PyTorch 기본 bias는 0(f = 0.5) → 직접 1~2로 올려 f ≈ 0.9로 시작하는 게 관례.

</details>

<details>
<summary>Q5. 그럼 LSTM은 기울기 소실을 "해결"했나?</summary>

**"크게 늦춘" 것.** forget gate가 1일 때만 곱이 없다. 진짜 강점은 **"언제 열고 닫을지를 학습으로 정한다"** 는 것. RNN엔 그 손잡이 자체가 없었다.

</details>

### 4. 변화 — 필요한 정보를 오래 유지, 먼 과거까지 학습

- 기울기 소실을 완전히 제거하지는 않지만 **크게 완화**하여 먼 과거의 정보를 학습하기 쉬워졌다.

**한눈에 정리**

| | |
| --- | --- |
| **문제** | RNN은 **먼 과거의 정보를 학습하기 어렵다** |
| **원인** | 기울기 **소실** 또는 **폭발** → 먼 과거까지 "고쳐!" 신호가 안 와서 **배우기 어렵다** (기억을 못 하는 게 아님) |
| **해결책** | 기억이 지나가는 별도 통로 **Cell state** + 얼마나 남기고 · 넣고 · 꺼낼지 정하는 **Gate 3개** |
| **변화** | 필요한 정보를 **오래 유지**하고, **먼 과거까지 학습**하기 쉬워짐 |

### 5. 수식 — 한 줄씩

![LSTM 수식을 한 줄씩 풀어 쓴 그림](/images/study-lstm-gru/lstm-gru-15-lstm-formula.svg)

**C = 이전 기억 × 얼마나 남길지 + 새 내용 × 얼마나 넣을지**

### 6. 코드 — 실제로는 이렇게 생겼다

**우선순위 : 설정값과 출력을 보자.**

```python
class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=32, batch_first=True)
        self.fc   = nn.Linear(32, 2)                # 기억 크기 → 클래스 수

    def forward(self, x):                  # x : (32, 20, 10)
        output, (h_n, c_n) = self.lstm(x)  # 순서대로 읽기(for 문)는 nn.LSTM 안에
        return self.fc(h_n[-1])            # 다 읽은 뒤의 마지막 기억 하나만 분류
```

- `input_size=10` → 한 시점에 특징 몇 개?
- `hidden_size=32` → 기억을 숫자 몇 개로 표현할까?
- `batch_first=True` → 숫자는 아니지만 입력 shape을 `(batch, seq, feature)`로 쓸까?

**shape 읽기 — 숫자 3개**

```
x        ( 32, 20, 10 )
           │   │    └─ 단어 1개를 표현하는 특징 10개        (= feature)
           │   └─────── 문장 1개에 단어 20개                 (= sequence length)
           └─────────── 한 번에 문장 32개 처리              (= batch size)

output   ( 32, 20, 32 )
           │   │    └─ 단어 하나 읽을 때마다 만든 기억 h 32개  (= hidden_size)
           │   └─────── 20개 단어 각각의 h 를 모두 보관
           └─────────── 32개 문장 각각에 대해

h_n      ( 1, 32, 32 )
           │   │    └─ 문장을 끝까지 읽은 뒤 남은 최종 기억 h 32개  (= hidden_size)
           │   └─────── 32개 문장 각각의 최종 기억
           └─────────── LSTM 층 수 × 방향 수   (= num_layers × directions)  1층 × 단방향
```

**함정 4개**

<details>
<summary>함정 1. <code>x</code>의 shape <code>(32, 20, 10)</code>에서 input_size가 뜻하는 것은?</summary>

**10 — 한 단어의 feature 수.** shape 맨 뒤 숫자.

</details>

<details>
<summary>함정 2. 분류할 때 <code>output</code> 전체를 <code>fc</code>에 넣으면?</summary>

- ✗ `self.fc(output)` — 모든 시점의 h를 다 던짐. 모양도 안 맞고 의미도 섞인다.
- ✓ `self.fc(h_n[-1])` — 문장을 다 읽은 뒤 **마지막 기억 하나**만.

</details>

<details>
<summary>함정 3. <code>h_n[-1]</code>과 <code>output[:, -1, :]</code>은 항상 같은가?</summary>

둘 다 "마지막 h"지만 **기준이 다르다.**

```
output[:, -1, :]   틀의 마지막 칸 (시간축 -1)
h_n[-1]            마지막 층의 최종 h (층축 -1)
```

| | 같은가 | 한 줄 이유 |
|---|:-:|---|
| 단방향 · 1층 · padding 없음 | ✓ | 마지막 칸 h = 최종 h. 같은 값 |
| padding 있음 | ✗ | 둘 다 **빈 칸을 지난 값.** 진짜 마지막은 `output[idx, lens-1]` |

```
padding 예 — 문장 3칸, 틀 6칸
[h1][h2][h3][ ][ ][ ]
         ↑          ↑
   진짜 마지막      -1 이 읽는 자리
```

</details>

<details>
<summary>함정 4. <code>batch_first</code>를 바꾸면 <code>hidden</code>의 축 순서도 바뀌나?</summary>

**`out`만 바뀐다.** `(seq, batch, hidden)` ↔ `(batch, seq, hidden)`으로 축 순서가 바뀐다.
`hidden`은 **항상 `(층 수, batch, hidden)`** — `batch_first`와 무관.

</details>

---

## ③ GRU

### 1. 문제 — LSTM은 파라미터가 많고 계산량이 크며 구조가 복잡하다

### 2. 원인 — 상태 2개(C, h)를 따로 관리하고 Gate 3개를 매 시점 계산한다

**잊을 양 계산 → 새로 넣을 양 계산 → 새 기억 후보 계산 → 출력할 양 계산**

"장기기억을 위해 정말 C와 h를 따로 관리하고, gate도 3개나 써야 할까?"

### 3. 해결책 — 상태 2개 → 1개, Gate 3개 → 2개

| | LSTM | | GRU |
|:-:|:-:|:-:|:-:|
| 상태 | `C_t` + `h_t` | → | **`h_t` 하나** |
| Gate | Forget + Input + Output | → | **Update + Reset** |

| | 직접 보존되는 것 | 식 |
|:-:|---|---|
| **LSTM** | Cell state | `C_(t-1) ──( × f_t )──▶ C_t` |
| **GRU** | Hidden state 자체 | `h_(t-1) ──( × (1 − z_t) )──▶ h_t` |

<details>
<summary>Q6. GRU의 z가 크면 옛것을 많이 남기나?</summary>

반대. **z는 "새것" 비율**, 옛것은 (1−z). r은 **새것을 만들 때** 참고 비율, z는 **섞을 때** 비율.

</details>

<details>
<summary>Q7. GRU는 LSTM의 성능 문제를 해결하였다 (O, X)?</summary>

**X.** "LSTM의 성능 문제 해결"이 아니라 **"LSTM의 구조적 복잡성 완화".** 성능은 task마다 다르다.

</details>

### 4. 변화 — 가볍고 빠르다

| LSTM의 아쉬운 점 | GRU에서의 변화 | 그 결과 |
| --- | --- | --- |
| 기억 상태가 `C_t`, `h_t` **2개** | `h_t` **1개로 통합** | 관리할 상태가 단순해짐 |
| Gate가 **3개** | Gate **2개** | 계산량 감소 |
| Forget/Input gate가 따로 결정 | **Update gate 하나로 통합** | 이전 기억과 새 기억의 비율을 한 번에 결정 |
| Output gate로 `C_t`에서 `h_t`를 따로 만듦 | **Output gate 제거** | 구조 단순화 |

**한눈에 정리**

| | |
| --- | --- |
| **문제** | LSTM은 **복잡하고 무겁다** — 상태 2 + Gate 3 = 파라미터 수와 계산량 증가 |
| **원인** | 매 시점 **C와 h 두 상태**를 따로 갱신하고, Gate 3개 + 새 후보 = **계산 네 벌** → 파라미터 · 시간이 늘어난다 |
| **해결책** | **상태 1 + Gate 2** — 남기기·넣기를 Gate 하나(z)로, C와 h를 합침 |
| **변화** | 같은 목적을 **더 간단하게.** 가볍고 빠르다 |

### 5. 수식 — 한 줄씩

![GRU 수식을 한 줄씩 풀어 쓴 그림](/images/study-lstm-gru/lstm-gru-16-gru-formula.svg)

### 6. 코드 — LSTM 코드에서 글자 몇 개만 바뀐다

**LSTM 코드에서 `(h_n, c_n)` → `h_n`.** 나머지(shape · batch_first · `h_n[-1]`)는 전부 같다.

```python
class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=10, hidden_size=32, batch_first=True)
        self.fc  = nn.Linear(32, 2)

    def forward(self, x):                  # x : (32, 20, 10)  — LSTM 과 같음
        output, h_n = self.gru(x)          # c_n 없음 ← 유일한 차이
        return self.fc(h_n[-1])
```

**반환값 비교**

```python
# LSTM
output, (h_n, c_n) = lstm(x)     # 파이프 c 와 출력 h 두 줄
# GRU
output, h_n        = gru(x)      # h 한 줄
```

---

## 지금 배운 아이디어가 Transformer 안에 그대로

![LSTM과 GRU에서 배운 gate와 덧셈 경로가 Transformer의 attention과 residual로 이어진다](/images/study-lstm-gru/lstm-gru-00-why.svg)

| 지금 배운 것 | Transformer 안에서 |
|---|---|
| **Gate** — 얼마나 흘려보낼지 0~1 | **Attention** — 누구를 얼마나 볼지 0~1 |
| **C의 더하기** — 곱셈 대신 더하기로 신호를 멀리 | **Residual** — `x + Layer(x)` |
| C 하나에 다 담는 한계 | Attention — 압축 없이 모두를 직접 봄 |

---

## 요약

![RNN, LSTM, GRU 요약](/images/study-lstm-gru/lstm-gru-09-summary.svg)

```
RNN → LSTM → GRU → [Seq2Seq] → Attention → Transformer
기억   오래 기억  가볍게   문장 변환    찾아보기     RNN 제거
```

```
코드 볼 때
① 입력 shape      (batch, seq, feature)
② input_size      feature 수 (seq 길이 아님)
③ hidden_size     기억 크기 = output 마지막 차원
④ output shape    (batch, seq, hidden × 방향 수)
⑤ 마지막 상태     h_n[-1] / output[:, -1] / output[idx, lens-1]
```

---

## 발표 후 질문

### 기억 사이즈(`hidden_size`)를 늘리면 더 좋아지나?

**경우에 따라. 단, 기울기 소실은 hidden을 키워도 안 풀린다.**

**1. 표현력은 늘어난다** — 한 시점에 담는 숫자가 32 → 64개면 더 복잡한 패턴을 담을 수 있다. 데이터가 충분하면 도움이 된다.

**2. 먼 과거 문제는 그대로** — 소실은 "메모가 작아서"가 아니라 **신호가 곱셈을 100번 거치며 사라지는 구조** 때문. 메모를 키워도 곱셈 횟수는 같다.
수업 실측 : RNN hidden 64 → 128(파라미터 3.5배)로 키웠더니 82.5% → **83.2%**, 겨우 +0.7%p. 같은 파라미터의 LSTM은 90.6%.
→ 구조(Gate)를 바꿔야 8%p, 크기를 키워선 1%p도 안 된다.

**3. 비용** — 파라미터·계산량이 늘고, 데이터가 적으면 **과적합**. 시간도 늘어난다.

**정리** : hidden은 "얼마나 자세히 기억할까", Gate는 "얼마나 멀리 기억할까". 지금 문제가 **멀리**라면 hidden이 아니라 구조. 얼마가 적당한지는 실험(32 · 64 · 128)으로 정한다.
