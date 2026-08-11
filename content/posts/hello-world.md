+++
title = '첫 글'
date = 2026-08-11T00:00:00+09:00
draft = false
slug = 'hello-world'
description = 'Hugo + PaperMod로 블로그를 열었다.'
tags = ['blog']
categories = ['일반']
+++

블로그 첫 글.

## 마크다운 기본

**굵게**, *기울임*, `인라인 코드`.

> 인용문은 이렇게.

- 목록
- 이렇게

1. 번호 목록
2. 이렇게

| 표 | 이렇게 |
|---|---|
| 값1 | 값2 |

```python
print("코드 블록")
```

## 새 글 쓰는 법

```
hugo new content posts/글-이름.md
```

`content/posts/글-이름.md` 가 생김. 위쪽 `draft = true` 를 `false` 로 바꾸면 공개됨.
