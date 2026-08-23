+++
title = '위클리페이퍼2'
date = 2026-08-23T17:00:00+09:00
draft = false
slug = 'weekly-paper-2'
description = 'loss가 NaN이 되는 진짜 이유와 원인별 감별법, 그리고 만족도 1~5점을 회귀로 풀지 분류로 풀지. 둘 다 직접 재현해서 숫자로 확인했다.'
tags = ['부트캠프', '위클리페이퍼', '딥러닝', '디버깅', '회귀', '분류', 'scikit-learn']
categories = ['개발']
+++

두 문제 모두 말로만 답하면 그럴듯한 목록이 되어버린다. 그래서 직접 재현하고, 고쳐보고, 숫자를 확인했다.
아래 코드는 전부 NumPy와 scikit-learn만 쓴다. 복사하면 그대로 돌아간다.

---

## Q1. loss가 줄다가 커지다가 NaN

> `model.fit()`으로 학습을 돌렸는데, loss 값이 처음엔 잘 줄어들다가 어느 순간부터 점점 커지더니 결국 NaN이 되어버렸어요. 어떤 원인들을 의심해볼 건가요? 각 원인을 어떻게 확인할 건가요?

### 먼저: NaN은 사인(死因)이 아니라 사망진단서다

NaN은 "숫자가 아님(Not a Number)"이라는 뜻이다. 여기서 중요한 건 **NaN이 원인이 아니라 결과**라는 점이다.

컴퓨터가 담을 수 있는 숫자에는 한계가 있다. 파이썬이 기본으로 쓰는 float64는 약 `1.8e308`(1.8 뒤에 0이 308개)까지만 표현한다. 이 한계를 넘으면 숫자가 아니라 `inf`(무한대)가 된다.

그리고 `inf`끼리 빼거나(`inf - inf`), 0을 곱하면(`0 × inf`) — 답을 정할 수 없으니 그때 **NaN**이 태어난다.

![NaN이 생기는 3단계. loss가 커짐 → float64 한계 1.8e308을 넘어 inf가 됨 → inf끼리의 연산에서 NaN 발생. NaN은 마지막 단계이므로 원인은 그 앞에 있다](/images/weekly-paper-2/q1-nan-cascade.svg)

즉 NaN을 본 시점은 이미 **한참 늦은 시점**이다. 진범은 "왜 loss가 그렇게까지 커졌는가"에 있다.

### 왜 커지는가 — 눈 감고 산 내려가기

모델 학습은 눈을 감고 산을 내려가는 것과 같다. 발밑의 기울기만 만져보고, 낮아지는 쪽으로 한 걸음 옮긴다. 이때 **보폭**이 learning rate(학습률)다.

보폭이 적당하면 골짜기 바닥에 잘 도착한다. 그런데 보폭이 너무 크면?

![학습률과 보폭. 적당한 보폭은 골짜기 바닥으로 수렴하지만, 너무 큰 보폭은 골짜기를 건너뛰어 반대편 더 높은 곳에 착지하고, 그러면 기울기가 더 커져서 다음 걸음은 더 멀리 튄다](/images/weekly-paper-2/q1-lr-overshoot.svg)

골짜기를 **건너뛰어** 반대편 벽에 착지한다. 반대편은 더 높으니 기울기가 더 가파르다. 기울기가 가파르면 다음 걸음은 더 크다. 더 크게 튀면 더 높은 곳에 착지한다. 이게 계속 반복된다.

한 번 튈 때마다 조금씩 커지는 게 아니라 **곱절로** 커진다. 그래서 몇 걸음 만에 1e308을 넘어버린다.

### 실제로 재현해보자

2층 신경망을 NumPy로 직접 짜서 학습률만 0.35로 올렸다. 나머지는 전부 정상이다.

```python
import numpy as np

def data(n=500, seed=0, scale=1.0, bad=False, outlier=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 3))
    y = (2*X[:,0] - X[:,1] + .5*X[:,2] + rng.normal(0, .1, n)).reshape(-1, 1)
    X[:,0] *= scale                  # 특정 열만 스케일 키우기
    if bad:     y[137,0] = np.nan    # 라벨에 결측 심기
    if outlier: y[137,0] = 1e9       # 라벨에 이상치 심기
    return X, y

class MLP:
    def __init__(s, d=3, h=16, seed=0):
        r = np.random.default_rng(seed)
        s.W1 = r.normal(0, np.sqrt(2/d), (d, h)); s.b1 = np.zeros(h)
        s.W2 = r.normal(0, np.sqrt(2/h), (h, 1)); s.b2 = np.zeros(1)

    def fit(s, X, y, lr=.01, epochs=3, bs=32):
        n = len(X); step = 0
        for ep in range(epochs):
            idx = np.random.default_rng(ep).permutation(n)
            for st in range(0, n, bs):
                b = idx[st:st+bs]; xb, yb = X[b], y[b]

                # 순전파
                z1 = xb @ s.W1 + s.b1; a1 = np.maximum(z1, 0)
                out = a1 @ s.W2 + s.b2; err = out - yb
                loss = np.mean(err**2)

                # 역전파
                dW2 = a1.T @ err * (2/len(b)); db2 = err.mean(0) * 2
                da1 = err @ s.W2.T * (2/len(b)) * (z1 > 0)
                dW1 = xb.T @ da1; db1 = da1.sum(0)

                # 기울기 크기와 가중치 크기를 같이 기록한다 (이게 진단의 핵심)
                g = np.sqrt(sum((q**2).sum() for q in (dW1, db1, dW2, db2)))
                s.W1 -= lr*dW1; s.b1 -= lr*db1; s.W2 -= lr*dW2; s.b2 -= lr*db2
                w = np.sqrt((s.W1**2).sum() + (s.W2**2).sum())

                print(f"step{step:>4} loss={loss:>12.4g}  |grad|={g:>12.4g}  |W|={w:>12.4g}")
                if not np.isfinite(loss):
                    print(f">>> step {step}: NaN 발생, 중단"); return
                step += 1

X, y = data()
MLP().fit(X, y, lr=0.35)
```

결과:

```text
step   0 loss=       6.087  |grad|=       13.39  |W|=       6.743
step   1 loss=         160  |grad|=       139.1  |W|=       37.27
step   2 loss=   5.244e+04  |grad|=    1.03e+04  |W|=        3298
step   3 loss=   7.901e+12  |grad|=   1.423e+10  |W|=   4.802e+09
step   4 loss=   1.603e+37  |grad|=   2.495e+28  |W|=   7.641e+27
step   5 loss=  9.101e+109  |grad|=    8.44e+82  |W|=   2.822e+82
step   6 loss=         inf  |grad|=         inf  |W|=         inf
>>> step 6: NaN 발생, 중단
```

중간에 `RuntimeWarning: overflow encountered` 경고가 같이 뜬다. 이게 바로 위에서 말한 2단계, 즉 `inf`가 만들어지는 순간이다. NumPy는 친절하게도 진범이 활동하는 시점을 알려주고 있었던 셈이다.

여섯 걸음 만에 죽는다. 그리고 loss만 커지는 게 아니라 **가중치 크기 `|W|`도 같이 폭증**한다. 이게 발산의 지문이다.

### 여기서 가장 중요한 한 가지: 로그를 step 단위로 바꿔라

위 표에서 계단이 선명하게 보이는 이유는 **배치(step)마다 찍었기 때문**이다. 만약 epoch마다 찍었다면 이렇게 보인다.

```text
ep 0 loss=nan     <- 끝. 아무 정보도 없다
```

한 epoch 안에는 배치가 수십 개 들어 있다. epoch 단위 로그는 그 사이에 벌어진 일을 전부 삼켜버린다. 그래서 "갑자기 NaN이 됐다"처럼 보이는 것이다.

Keras라면 이렇게 바꾼다.

```python
import tensorflow as tf

log = tf.keras.callbacks.LambdaCallback(
    on_batch_end=lambda batch, logs: print(batch, logs['loss'])
)
model.fit(X, y, callbacks=[log])
```

**이 한 줄이 아래 감별표를 쓸 수 있게 해준다.** 이걸 안 하면 원인 다섯 개를 구분할 방법이 없다.

### 원인 5가지 — 각각 지문이 다르다

![원인별 loss 곡선 지문 세 가지. 발산형은 계단처럼 곱절로 커지고, 즉사형은 정상이다가 한 번에 NaN이 되고, 지뢰형은 평온하다가 특정 배치에서 한 번에 폭발한다](/images/weekly-paper-2/q1-fingerprints.svg)

| # | 원인 | 지문 | 확인 방법 |
|---|---|---|---|
| ① | **학습률이 너무 큼** | loss가 계단식으로 곱절 증가. 가중치 크기도 같이 폭증. 매번 재현됨 | 학습률만 1/10로 낮춰본다 |
| ② | **입력 정규화 안 함** | ①과 똑같아 보이지만 **step 0부터 loss가 1e5** | 열별 최댓값 비율 확인 |
| ③ | **데이터에 NaN/inf 있음** | **커지는 구간이 없다.** 정상 → 바로 NaN | `np.isnan(X).sum()` |
| ④ | **라벨에 이상치** | 평온하다가 그 배치를 만나는 순간 1e60 | `y`의 백분위수 확인 |
| ⑤ | **log(0) 또는 0으로 나눔** | cross entropy, `sqrt`, `/std`를 쓸 때만 | 해당 연산에 작은 값(eps) 더하기 |

질문의 "**점점 커지다가**"라는 표현이 이미 단서다. ③은 점점 커지지 않고 그냥 죽는다. 그러니 ①②④가 후보로 좁혀진다.

③과 ④는 실제로 이렇게 구분된다.

```text
[라벨 1개가 NaN]                 [라벨 1개가 1e9]
step 3 loss=  3.992              step 3 loss=  3.992
step 4 loss=    nan   <- 즉사     step 6 loss= 3.96e+60   <- 폭발
                                 step 8 loss=      inf
```

②는 왜 ①과 같아 보일까? **입력값이 크면 학습률을 안 건드려도 실제 이동 거리가 커지기 때문**이다. 기울기는 입력값에 비례하니, 입력이 1000배면 걸음도 1000배가 된다. 학습률 0.01이 사실상 10인 셈이다.

```text
[입력 한 열만 1000배, 학습률은 0.01 그대로]
step 0 loss=  1.072e+05   <- 첫 걸음부터 이 값
step 1 loss=  9.913e+20
step 3 loss= 1.781e+206
step 4 loss=        inf
```

### 확인 순서 — fit() 돌리기 전 5분

원인의 절반은 학습을 시작하기도 전에 잡을 수 있다.

```python
def preflight(X, y):
    print("[1] 데이터에 NaN/inf 있나")
    for nm, A in (("X", X), ("y", y)):
        nn, ni = np.isnan(A).sum(), np.isinf(A).sum()
        print(f"    {'FAIL' if nn+ni else 'ok  '} {nm}: NaN={nn} inf={ni}")

    print("[2] 열별 스케일 (최댓값 비율이 100을 넘으면 정규화 필수)")
    mx = np.abs(X).max(0)
    for j in range(X.shape[1]):
        print(f"    col{j}: mean={X[:,j].mean():>10.3g} "
              f"std={X[:,j].std():>10.3g} absmax={mx[j]:>10.3g}")
    r = mx.max() / max(mx.min(), 1e-12)
    print(f"    -> 비율 {r:.3g} {'FAIL 정규화 하라' if r > 100 else 'ok'}")

    print("[3] 타깃에 이상치 있나")
    q = np.percentile(y[np.isfinite(y)], [0, 50, 99, 100])
    print(f"    min={q[0]:.4g} med={q[1]:.4g} p99={q[2]:.4g} max={q[3]:.4g}")
    print("    FAIL: 이상치 의심" if q[3] > q[2]*100 else "    ok")

preflight(*data(scale=1e3, outlier=True))
```

```text
[1] ok   X: NaN=0 inf=0 / ok   y: NaN=0 inf=0
[2] col0: mean=      -126 std=       979 absmax=  3.77e+03
    col1: mean=    0.0798 std=     0.957 absmax=       3.9
    col2: mean=  0.000367 std=      1.02 absmax=      2.97
    -> 비율 1.27e+03 FAIL 정규화 하라
[3] min=-7.831 med=-0.2568 p99=4.621 max=1e+09
    FAIL: 이상치 의심
```

심어둔 원인 두 개가 학습을 시작하기도 전에 이름이 나왔다.

### 그다음: 학습률만 바꿔가며 돌린다

데이터가 깨끗한데도 죽으면, **다른 건 전부 고정하고 학습률만** 바꾼다. 변수를 하나만 움직여야 원인이 특정된다.

```text
lr=1        발산(NaN)
lr=0.35     발산(NaN)
lr=0.1      final_mse=0.03028   <- 안전 상한
lr=0.03     final_mse=0.0644
lr=0.01     final_mse=0.2362
lr=0.003    final_mse=1.555     <- 너무 느려서 덜 배웠다
```

살아나면 ①②로 확정. 안 살아나면 데이터 문제(③④⑤)다.

그래도 모르겠으면 **배치 하나만 골라 그것만 반복 학습**시켜본다. 그것마저 NaN이면 모델 구조나 손실 함수 문제이고, 잘 되면 특정 배치가 범인이니 배치별 loss를 찍어 찾아낸다.

### 처방 — 효과 순서대로

1. **입력 표준화** (`(X - 평균) / 표준편차`). 원인 ②를 직격한다. 위의 1000배 데이터도 표준화만으로 정상 수렴했다
2. **학습률 1/10.** 가장 싸고 빠른 확인이자 처방
3. **gradient clipping** (`clipnorm=1.0`). 걸음이 아무리 커도 최대 길이를 제한한다. 단, 이건 **응급조치**다 — 실제로 학습률 1.5에 clipping을 걸었더니 NaN은 안 났지만 loss가 0.15~3.7 사이를 계속 요동쳤다. **안 죽었을 뿐 학습이 안 되고 있다**
4. **작은 값 더하기.** `log(p + 1e-7)`, `x / (std + 1e-8)`. 원인 ⑤ 전용이다
5. **fp16(half precision)을 쓰는 중이라면** loss scaling을 켠다. fp16의 한계는 65504라서 훨씬 빨리 터진다

### 한 줄 요약

**NaN을 보면 NaN을 찾지 말고, 그 앞에서 loss가 몇 배씩 커진 지점을 찾아라. 그러려면 로그부터 step 단위로 바꿔야 한다.**

---

## Q2. 만족도 1~5점, 회귀인가 분류인가

> 고객 만족도 설문 결과(1점~5점)를 예측하는 모델을 만들어야 해요. 이 문제는 회귀로 풀 수도 있고, 5개 범주에 대한 분류로 풀 수도 있어요. 어떤 방식을 선택할 건가요? 각 방식의 장단점을 들어 설명해 주세요.

### 문제의 핵심 — 별점은 숫자인가 이름표인가

이 질문이 어려운 이유는 1~5점이 **숫자와 이름표의 중간**에 있기 때문이다.

- **회귀로 푼다** = 1~5를 완전한 숫자로 본다. 순서가 있고, 간격도 일정하다고 가정한다
- **분류로 푼다** = 1~5를 이름표로 본다. '1점'과 '5점'은 '사과'와 '자동차'처럼 아무 관계 없는 다른 이름이다

그런데 진실은 둘 다 아니다.

- 순서는 **있다**. 4점은 3점보다 확실히 좋다 → 분류의 가정은 틀렸다
- 간격은 **일정하지 않다**. 보통 사람은 웬만하면 4점을 준다. 5점을 주려면 훨씬 감동해야 한다. 즉 4→5의 문턱이 2→3의 문턱보다 높다 → 회귀의 가정도 틀렸다

![별점의 성질. 회귀는 순서와 등간격을 모두 가정하고 분류는 둘 다 없다고 가정하지만, 실제 별점은 순서는 있고 간격은 일정하지 않아 둘 사이에 있다](/images/weekly-paper-2/q2-ordinal-nature.svg)

이런 데이터를 **순서형(ordinal)** 이라고 부른다. 회귀와 분류 사이에 답이 있다.

### 실험 데이터 만들기

실제 설문의 두 가지 성질을 일부러 넣었다. (1) 4·5점에 몰리는 편향, (2) 4→5 문턱이 유독 높은 비등간격.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, confusion_matrix

def make(n=4000, seed=0):
    r = np.random.default_rng(seed)
    X = np.column_stack([
        r.normal(0, 1, n),      # 대기시간 (길수록 불만)
        r.normal(0, 1, n),      # 직원 친절도
        r.integers(0, 2, n),    # 재방문 여부
        r.normal(0, 1, n),      # 가격 민감도
    ])
    latent = -1.2*X[:,0] + 1.5*X[:,1] + 0.8*X[:,2] - 0.5*X[:,3] + r.normal(0, 1.0, n)
    cuts = [-1.8, -0.7, 0.3, 2.2]      # 마지막 간격이 넓다 = 5점은 드물다
    return X, np.digitize(latent, cuts) + 1

X, y = make(); Xte, yte = make(1500, seed=99)
print("클래스 분포(train):", {k: int((y == k).sum()) for k in range(1, 6)})
```

```text
클래스 분포(train): {1: 665, 2: 564, 3: 646, 4: 1210, 5: 915}
```

### 다섯 가지 방식을 붙여봤다

```python
def report(name, pred):
    print(f"{name:<22} MAE={mean_absolute_error(yte, pred):.4f}  "
          f"정확도={(pred == yte).mean():.3f}  "
          f"±1내={(np.abs(pred - yte) <= 1).mean():.3f}  "
          f"QWK={cohen_kappa_score(yte, pred, weights='quadratic'):.4f}")

rnd = lambda v: np.clip(np.round(v), 1, 5).astype(int)

reg = GBR(random_state=0).fit(X, y)
report("A. 회귀+반올림", rnd(reg.predict(Xte)))

clf = GBC(random_state=0).fit(X, y)
P = clf.predict_proba(Xte)
report("B. 분류+argmax", clf.predict(Xte).astype(int))
report("C. 분류+기댓값", rnd(P @ clf.classes_))     # 확률 × 점수의 합
```

순서형은 이렇게 만든다. "몇 점인가?"를 한 번에 맞히는 대신, **"1점보다 높은가?" "2점보다 높은가?" … 를 네 번 물어본다.**

```python
def ordinal(X, y, Xte):
    cum = []
    for k in range(1, 5):
        m = GBC(random_state=0).fit(X, (y > k).astype(int))   # P(y > k)
        cum.append(m.predict_proba(Xte)[:, 1])
    cum = np.array(cum)
    cum = np.minimum.accumulate(cum, axis=0)   # 단조성 강제 (필수)
    # 누적확률의 차이 = 각 점수의 확률
    return np.vstack([1-cum[0], cum[0]-cum[1], cum[1]-cum[2],
                      cum[2]-cum[3], cum[3]]).T

Po = ordinal(X, y, Xte)
report("D. 순서형+argmax", (Po.argmax(1) + 1).astype(int))
report("E. 순서형+기댓값", rnd(Po @ np.arange(1, 6)))
```

![순서형 모델의 구조. 5개 중 하나를 고르는 대신 1점 초과인가, 2점 초과인가, 3점 초과인가, 4점 초과인가를 각각 예측한 뒤 누적확률의 차이로 각 점수의 확률을 구한다](/images/weekly-paper-2/q2-ordinal-method.svg)

### 결과

```text
A. 회귀+반올림          MAE=0.4680  정확도=0.581  ±1내=0.953  QWK=0.8394
B. 분류+argmax         MAE=0.4973  정확도=0.588  ±1내=0.924  QWK=0.8250
C. 분류+기댓값          MAE=0.4793  정확도=0.573  ±1내=0.949  QWK=0.8312
D. 순서형+argmax        MAE=0.4787  정확도=0.589  ±1내=0.939  QWK=0.8394
E. 순서형+기댓값         MAE=0.4613  정확도=0.593  ±1내=0.947  QWK=0.8394  <- 전 지표 1등
```

여기서 함정 하나. **정확도만 보면 분류(0.588)가 회귀(0.581)보다 높다.** 그런데 나머지 세 지표는 전부 회귀가 이긴다. 왜 그럴까?

### 정확도는 "얼마나 크게 틀렸는지"를 못 본다

실제 5점인 고객을 4점이라 예측한 것과, 1점이라 예측한 것. 정확도는 **둘 다 똑같이 오답 1개**로 센다. 하지만 현실에서 이 둘은 전혀 다르다.

크게 틀린 비율을 재보면 이렇다.

```text
회귀          2칸 이상 틀림 : 4.67%   3칸 이상 : 0.20%
분류 argmax    2칸 이상 틀림 : 7.60%   3칸 이상 : 0.93%   <- 4.6배
순서형+기댓값   2칸 이상 틀림 : 5.27%   3칸 이상 : 0.20%
```

혼동 행렬(행=실제, 열=예측)을 보면 더 선명하다.

```text
        분류                      회귀
[[187  43  21   9   0]     [[140  90  27   3   0]
 [ 58  80  38  37   1]      [ 19 109  70  16   0]
 [ 15  55  57 115   5]      [  3  62 110  72   0]
 [  4  19  44 311  64]      [  0  16 101 283  42]
 [  0   0   3  87 247]]     [  0   0   5 103 229]]
```

분류는 실제 2점을 5점이라 한 케이스가 1건, 3점을 5점이라 한 게 5건 있다. 회귀는 대각선에서 2칸 이상 벗어난 칸이 거의 전부 0이다.

이유는 **손실 함수**에 있다. 분류가 쓰는 cross entropy는 "1점을 2점이라 함"과 "1점을 5점이라 함"에 **똑같은 벌점**을 준다. 순서 정보를 아예 안 쓰기 때문이다.

![분류와 회귀의 오답 벌점 차이. 실제 5점을 4점이라 예측하나 1점이라 예측하나 분류는 같은 벌점을 주지만 회귀는 거리에 비례해 벌점을 준다](/images/weekly-paper-2/q2-penalty.svg)

### 그래서 평가 지표는 QWK를 쓴다

정확도가 순서를 무시한다면, 순서를 아는 지표를 써야 한다. **QWK(Quadratic Weighted Kappa)** 는 틀린 거리의 제곱만큼 감점하는 지표다. 1에 가까울수록 좋고, 0이면 찍는 것과 같다는 뜻이다.

```python
cohen_kappa_score(y_true, y_pred, weights='quadratic')
```

위 결과에서 정확도는 분류가 이겼지만 QWK는 회귀·순서형이 이겼다(0.8394 vs 0.8250). **어느 쪽이 실제로 쓸 만한지를 QWK가 제대로 말해준다.**

### 데이터가 적으면 회귀가 크게 이긴다

학습 데이터 크기만 바꿔가며 MAE를 재봤다(낮을수록 좋음).

```text
 n_train |       회귀   분류argmax    순서형+기댓값
     150 |   0.5867     0.6667      0.5920
     400 |   0.5220     0.5940      0.5167
    1000 |   0.4847     0.5500      0.4833
    4000 |   0.4680     0.4973      0.4613
   12000 |   0.4607     0.4780      0.4513
```

150개일 때 회귀 0.587 vs 분류 0.667로 14% 차이가 난다. 12000개가 되면 격차가 좁혀지지만 **역전되지는 않는다.**

이유는 간단하다. 회귀는 배울 게 한 세트인데, 분류는 클래스마다 따로 배워야 한다. 같은 데이터를 5등분해서 쓰는 셈이라 데이터에 목마르다.

### 반대로 희귀 클래스는 분류가 낫다

5점을 전체의 2.4%로 줄여봤다.

```text
회귀: 5점이라 예측한 개수 20개   recall=0.217  precision=0.650
분류: 5점이라 예측한 개수 38개   recall=0.300  precision=0.474
```

회귀는 애매하면 평균 쪽으로 답을 내는 성질이 있어서 극단값을 잘 안 뱉는다. **"드문 극단값 자체를 찾아내는 것"이 목적이라면 분류 + `class_weight`가 낫다.**

### 그런데 진짜 질문은 "이걸로 뭘 할 건가"다

실무에서 "3.7점이냐 4.0점이냐"가 정확히 필요한 경우는 드물다. 보통은 **"불만 고객을 먼저 찾아 연락한다"** 같은 결정에 쓴다.

그래서 목적을 바꿔서 재봤다. 1500명 중 위험도 상위 10%(150명)에게 연락한다면, 그 안에 실제 불만 고객(1~2점)이 몇 명이나 들어 있을까?

```python
score_reg = -reg.predict(Xte)              # 예측 점수가 낮을수록 위험
score_clf = P[:, :2].sum(1)                # P(1점) + P(2점)을 직접 사용

for name, s in [("회귀 점수 기준", score_reg), ("분류 P(1)+P(2) 기준", score_clf)]:
    top = np.argsort(-s)[:150]
    print(f"{name:<22} 실제 불만 {(yte[top] <= 2).sum()}명  "
          f"precision={(yte[top] <= 2).mean():.3f}")
```

```text
회귀 점수 기준           상위 150명 중 실제 불만 147명   precision=0.980
분류 P(1)+P(2) 기준     상위 150명 중 실제 불만 148명   precision=0.987
(무작위로 뽑았을 때 = 0.316)
```

**차이가 사실상 없다.** 둘 다 무작위 대비 3배 넘게 잘 맞힌다.

즉 회귀냐 분류냐 하는 논쟁은 **정확한 점수 값 자체가 필요할 때만** 의미가 있다. 순위만 필요하면 어느 쪽을 골라도 된다.

### 장단점 정리

| | 회귀 | 분류 |
|---|---|---|
| **가정** | 순서 O, 등간격 O | 순서 X, 등간격 X |
| **장점** | 순서·거리 정보를 그대로 사용<br>크게 틀리는 일이 적음<br>데이터가 적어도 잘 버팀<br>"3.7점" 같은 연속값이 나와 순위 매기기·기준선 조정이 쉬움 | 등간격 가정이 필요 없음<br>클래스별 확률이 나옴 → "1점 줄 확률 40%" 같은 의사결정에 직결<br>"3점만 유독 다른" 불규칙 패턴도 학습 가능 |
| **단점** | **등간격 가정이 실제로는 거짓**<br>평균 쪽으로 수축해서 1점·5점을 과소 예측<br>확신 정도를 표현 못 함 | **순서 정보를 버림** → 크게 틀림<br>클래스마다 파라미터가 필요해 데이터 갈증<br>희귀 클래스에 취약 |

### 실전 권장 5가지

1. **순서형(방식 E)을 기본으로.** 위 `ordinal()` 함수가 전부다. `np.minimum.accumulate`로 단조성을 강제하는 부분을 빠뜨리면 안 된다 — "3점 초과 확률"이 "2점 초과 확률"보다 커지는 모순이 생긴다
2. **평가 지표는 QWK.** 정확도는 쓰지 않는다. MAE는 차선책이다
3. **분류를 쓰더라도 argmax 말고 기댓값**(`확률 × 점수의 합`)으로 예측한다. 실측 MAE가 0.4973 → 0.4793으로 개선됐다. 순서 정보를 공짜로 되찾는 셈이다
4. **회귀 손실은 MSE보다 MAE나 Huber.** MSE는 1↔5 오차를 1↔2 오차의 16배로 벌해서 이상치에 끌려다닌다
5. **반올림 대신 임계값을 최적화한다.** 등간격이 아니니 `1.5 / 2.5 / 3.5 / 4.5`로 자를 이유가 없다. 학습 데이터에서 QWK가 최대가 되는 지점(예: `1.4 / 2.6 / 3.3 / 4.2`)을 찾아 자른다

### 한 줄 요약

**별점은 숫자도 이름표도 아니라 순서다. 그러니 회귀와 분류 중 고르지 말고 순서형을 쓰고, 정확도 대신 QWK로 채점하라.**

---

## 두 문제의 공통점

Q1은 "loss가 NaN이 됐다"는 관찰에서 멈추면 안 되고 **그 앞의 곡선**을 봐야 했다.
Q2는 "정확도가 더 높다"는 관찰에서 멈추면 안 되고 **어떻게 틀렸는지**를 봐야 했다.

둘 다 **눈에 보이는 마지막 숫자 하나**가 원인이나 성능을 대표하지 못한다는 이야기다.
그래서 Q1은 로그를 step 단위로 쪼갰고, Q2는 지표를 QWK로 바꿨다.
