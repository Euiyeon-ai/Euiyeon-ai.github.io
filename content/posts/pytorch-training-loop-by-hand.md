+++
title = 'PyTorch 학습 한 바퀴 — backward는 계산만, 값을 바꾸는 건 update'
date = 2026-09-07T17:00:00+09:00
draft = false
slug = 'pytorch-training-loop-by-hand'
description = 'y = 2x + 1을 모델이 스스로 찾아가는 과정을 손으로 따라간다. 예측 → loss → backward → update → zero_ 여섯 단계, 학습률 실험, 조용히 바뀌는 dtype까지.'
tags = ['부트캠프', '스터디 발표', '딥러닝', 'PyTorch']
categories = ['개발']
+++

부트캠프 스터디에서 발표한 자료입니다. PyTorch Day01 실습 P11~P15(update · loop · lr · 구조 찾기 · dtype)를 "학습 한 바퀴"라는 하나의 흐름으로 묶었습니다. 질문은 접어 두었으니 먼저 답을 예상해 보고 펼쳐 보세요.

---

## 상황

`y = 2x + 1` 이라는 직선을 모델이 스스로 찾아가게 한다.

- 모델이 아는 것: `prediction = w * x + b` 라는 모양
- 모델이 모르는 것: w와 b의 값 (정답은 w=2, b=1)
- 하는 일: 예측 → 틀린 만큼 재기 → 어느 쪽으로 고칠지 계산 → 조금 고치기 → 반복

---

## 1. P11 · 학습 한 바퀴를 손으로 (manual update)

x=2, 정답 5, w=1, b=0, lr=0.05로 **딱 한 번** 학습한다.

<details>
<summary>Q1. w=1, b=0, x=2, 정답 5일 때 첫 예측값과 loss는? (계산해보기)</summary>

예측 = 1×2+0 = **2**, loss = (2-5)² = **9**.

</details>

| 단계 | 코드 | 하는 일 | 이번 예시 값 |
| --- | --- | --- | --- |
| ① 예측 | `pred = w*x + b` | 지금 w,b로 찍어봄 | 1×2+0 = **2** |
| ② loss | `(pred - y)**2` | 얼마나 틀렸나 | (2-5)² = **9** |
| ③ backward | `loss.backward()` | w,b를 어느 쪽으로 움직여야 loss가 줄지 계산 | grad_w = **-12**, grad_b = **-6** |
| ④ update | `w -= lr * w.grad` | 그 방향으로 조금 이동 | w = 1+0.6 = **1.6**, b = **0.3** |
| ⑤ reset | `w.grad.zero_()` | 계산한 방향 지우기 | grad = **0** |
| ⑥ 다시 예측 | 새 w,b로 ① ② | 진짜 나아졌나 확인 | pred **3.5**, loss **2.25** |

loss가 9 → 2.25로 줄었다. 한 바퀴가 제대로 돌았다는 뜻.

![학습 한 바퀴: 예측 → 틀림 재기 → 방향 계산 → 조금 고치기 → 지우기 → 다시 예측으로 반복](/images/study-pytorch-day01/p11-one-step.svg)

<details>
<summary>Q2. <code>loss.backward()</code>를 실행하면 w와 b 값이 바뀔까?</summary>

**아니다.** backward는 gradient(어느 쪽으로 가야 하는지)를 **계산만** 한다. 값은 `w -= lr * w.grad` 같은 update 줄에서 바뀐다.

</details>

> **꼭 기억 · P11**
>
> - `backward()`는 **계산만** 한다. w,b 값은 안 바뀐다. 바꾸는 건 ④ update의 몫. → **backward와 update는 다른 줄, 다른 책임.**
> - update는 `with torch.no_grad():` 안에서 한다. "학습 계산"이 아니라 "값 수정"이라서 기록(추적)이 필요 없기 때문.
> - `zero_()`를 안 하면 다음 바퀴 gradient가 **이전 것에 더해진다.** 이름 끝 `_`는 "원본을 직접 바꾼다"는 PyTorch 관례.

<details>
<summary>Q3. update를 <code>with torch.no_grad():</code> 안에서 하는 이유는?</summary>

update는 "학습 계산"이 아니라 단순한 "값 수정"이다. 이 연산까지 PyTorch가 추적(기록)하면 불필요하고, 다음 backward가 꼬일 수 있어서 기록을 끈다.

</details>

<details>
<summary>Q4. <code>w.grad.zero_()</code>를 빼먹고 loop를 돌리면 어떻게 될까?</summary>

이번 바퀴 gradient가 이전 바퀴 gradient에 **더해진다**(누적). 매 바퀴 update 후 반드시 지운다.

</details>

<details>
<summary>Q5. <code>zero_()</code> 끝의 <code>_</code>는 무슨 뜻?</summary>

새 텐서를 만들지 않고 **원본을 직접 바꾼다**(in-place)는 PyTorch 관례.

</details>

---

## 2. P12 · 한 바퀴를 21번 반복 (manual linear regression)

P11의 여섯 단계를 `for` 안에 넣는다. 데이터는 네 점 (0,1),(1,3),(2,5),(3,7). w=b=0, lr=0.1.

```python
for step in range(21):
    pred = w * x + b                      # ① 예측
    assert pred.shape == y.shape          # 안전장치
    loss = ((pred - y)**2).mean()         # ② MSE
    loss.backward()                       # ③ 방향 계산
    with torch.no_grad():
        w -= lr * w.grad                  # ④ 이동
        b -= lr * b.grad
    w.grad.zero_(); b.grad.zero_()        # ⑤ 지우기
```

| step | w | b | loss |
|---|---|---|---|
| 0 | 0.00 | 0.00 | 21.0 |
| 5 | 2.02 | 0.96 | 0.0005 |
| 20 | 2.007 | 0.985 | 0.00008 |

- 5바퀴 만에 정답(2, 1) 근처에 도착. 이후는 미세 조정.
- P11과 달라진 점 두 개: 점이 여러 개라 loss에 `.mean()`을 붙이고(MSE), shape이 `[4,1]`로 맞는지 `assert`로 확인한다.

> **꼭 기억 · P12**
>
> 순서는 항상 **forward → loss → backward → update → reset**. 이 순서가 바뀌면 안 된다.
> P11의 한 바퀴를 그대로 `for` 안에 넣은 것. 새로운 개념 없음, 반복만 추가.

<details>
<summary>Q6. 학습 한 바퀴의 순서를 말해보자.</summary>

예측(forward) → loss → backward → update(no_grad) → reset(zero_). 이 순서로 반복.

</details>

---

## 3. P13 · 학습률(learning rate) 실험

같은 시작(w=b=0), 같은 데이터에서 lr만 0.01 / 0.1 / 0.5로 바꿔 **한 걸음**만 가본다.

<details>
<summary>Q7. lr을 0.01, 0.1, 0.5로 바꾸면 gradient 값도 서로 달라질까? (예상해보기)</summary>

**똑같다**(-17, -8). gradient는 데이터와 현재 위치가 정한다. lr은 그 방향으로 **얼마나** 갈지만 정한다.

</details>

| lr | grad_w | grad_b | 새 w | 새 b | 새 loss (시작 21.0) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | -17 | -8 | 0.17 | 0.08 | 17.6 → 거의 안 움직임 |
| 0.1 | -17 | -8 | 1.7 | 0.8 | 0.53 → 잘 감 |
| 0.5 | -17 | -8 | 8.5 | 4.0 | 215 → 정답을 지나쳐 튕겨나감 |

- 방향과 기울기 정보는 데이터와 현재 위치가 정한다. 그래서 gradient는 세 경우 똑같다.
- 실제로 **얼마나 움직일지**는 lr이 정한다.
- 너무 작으면 답답하고, 너무 크면 loss가 오히려 폭발한다(21 → 215).

<details>
<summary>Q8. lr=0.5에서 loss가 21 → 215로 커진 이유는?</summary>

보폭이 너무 커서 정답(최저점)을 **지나쳐** 반대편 더 높은 곳으로 튕겨나갔다. lr이 크면 오히려 발산할 수 있다.

</details>

> **꼭 기억 · P13**
>
> - **gradient = 방향, lr = 보폭.** 둘은 다른 역할.
> - 이 실험에서 0.1이 좋았다고 "lr은 0.1이 정답"이라고 말하면 안 된다. **데이터·모델마다 다르다. 보편적 best lr은 없다.**

<details>
<summary>Q9. "이 실험에서 lr=0.1이 제일 좋았으니 항상 0.1을 쓰면 된다." 맞을까?</summary>

**틀렸다.** 이 데이터·이 모델에서 좋았을 뿐이다. 문제마다 적절한 lr은 다르다. 보편적 best는 없다.

</details>

---

## 4. P14 · 처음 보는 코드에서 구조 찾기 (final retrieval)

변수 이름이 전부 낯설어도(`feature`, `scale`, `offset`, `estimate`, `distance`) 역할은 똑같다.

<details>
<summary>Q10. 아래 코드에서 "학습되는 값(parameter)"은 어느 줄일까? 가장 빨리 구별하는 방법은?</summary>

`scale`, `offset`. `requires_grad=True`가 붙어 있으면 parameter. 안 붙어 있으면 입력이나 정답.

</details>

| 코드 | 역할 |
|---|---|
| `feature = tensor([1.,3.])` | Tensor + **입력**(x) |
| `answer = tensor([4.,8.])` | Tensor + **정답**(y) |
| `scale = tensor(0.5, requires_grad=True)` | **parameter** w |
| `offset = tensor(1., requires_grad=True)` | **parameter** b |
| `estimate = scale*feature + offset` | **prediction** |
| `distance = ((estimate-answer)**2).mean()` | **loss** |
| `distance.backward()` | **gradient 계산** |
| `with no_grad(): scale -= 0.05*scale.grad` | **update** |
| `scale.grad.zero_()` | **reset** |

**Parameter는 학습하면서 수정해야 하므로, "어떻게 수정할지"를 알려주는 gradient가 필요하다 → 그래서 `requires_grad=True`.**

> **꼭 기억 · P14**
>
> - 구별법 한 줄: `requires_grad=True`가 붙은 건 **학습되는 값(parameter)**, 안 붙은 건 **주어진 값(입력·정답)**.
> - 이름이 뭐든 구조는 같다: **값 준비 → 예측 → 틀림 측정 → 방향 계산 → 값 수정 → 정리.**

---

## 5. P15 · dtype이 조용히 바뀐다 (자동 승격)

```python
labels = torch.tensor([0, 1, 2])        # int64
scores = torch.tensor([0.1, 0.8, 0.3])  # float32
total  = labels * scores                # → ?
```

<details>
<summary>Q11. <code>total</code>의 dtype은? 에러가 날까?</summary>

에러 없이 **float32**. int64가 float32로 자동 승격된다.

</details>

- 정수 × 실수 = 에러 없이 **실수(float32)** 로 자동 승격된다.
- 에러가 안 나서 오히려 위험하다. 내가 모르는 사이 dtype이 바뀔 수 있다. → 헷갈리면 `.dtype`을 찍어본다.

<details>
<summary>Q12. 그러면 분류 문제의 정답 라벨도 항상 <code>.float()</code>로 바꾸는 게 좋을까?</summary>

**아니다.** 분류 loss(예: CrossEntropyLoss)는 라벨이 **정수(int64)** 여야 한다. dtype은 "내가 쓰는 loss가 뭘 요구하는가"를 따른다. P15는 승격을 관찰하는 실습이지 처방이 아니다.

</details>

> **꼭 기억 · P15**
>
> - dtype은 **에러 없이 조용히** 바뀔 수 있다. 의심되면 `.dtype` 출력.
> - "라벨은 항상 `.float()`로"가 **아니다.** 분류 loss(예: CrossEntropyLoss)는 정답 라벨이 **정수**여야 한다. **dtype은 내가 쓰는 loss가 요구하는 계약을 따른다.**

---

## 한 장 요약

```
값 준비(Tensor) → 예측(w*x+b) → 틀림 측정(loss) → 방향 계산(backward)
→ 값 수정(update, no_grad 안에서) → 정리(zero_) → 반복
```

- backward ≠ update. 계산과 수정은 다른 줄.
- `zero_()` 빼먹으면 gradient가 누적된다.
- gradient는 방향, lr은 보폭. 보편적 best lr은 없다.
- dtype은 에러 없이 바뀔 수 있다. loss가 요구하는 dtype을 따른다.

---

## 발표 후 질문

### Q13. 정수가 실수로 바뀌면 왜 위험한가?

분류 라벨 0, 1, 2는 클래스 번호라서 정수여야 한다.
CrossEntropyLoss는 이 값을 "0번 클래스, 1번 클래스"처럼 인덱스로 사용하기 때문에 float이면 에러가 난다.

문제는 중간에 float으로 바뀌어도 바로 에러가 안 나고, 나중에 loss에서 에러가 나 원인을 찾기 어렵다는 것이다.

### Q14. 분류는 정수라면 회귀는 소수여도 되나?

된다. 회귀는 값의 크기를 예측하므로 보통 float을 쓴다.

- 회귀: 집값 3.7, 온도 21.5 → float
- 분류: 0=고양이, 1=개 → int

### Q15. pred와 y의 shape이 같은지 왜 확인하나?

shape이 다르면 broadcasting 때문에 원하지 않는 계산이 될 수 있다.

pred=[4,1], y=[4]이면 [4,4]가 되어, 원래는 예측 4개를 각각의 정답 4개와 1:1로 비교해야 하는데, 각 예측이 모든 정답과 비교된다.

그래서 잘못된 loss가 계산되는데도 에러 없이 학습이 진행될 수 있어 assert로 shape을 확인한다.
