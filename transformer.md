# Transformer

## 1. Transformer 개요 | Overview

### 전체 구조 | Overall Architecture

```
                        ┌─────────────────────────────────────────┐
                        │           TRANSFORMER                    │
      ENCODER           │                       DECODER            │
  ┌─────────────┐       │           ┌──────────────────────────┐   │
  │ Feed Forward│ ──────┼──────────▶│ Encoder-Decoder Attention│   │
  │             │       │           │ (Cross-Attention)        │   │
  │ Add & Norm  │       │           ├──────────────────────────┤   │
  │             │       │           │ Masked Self-Attention    │   │
  │ Self-Attention      │           │ (Causal Attention)       │   │
  └──────┬──────┘       │           └──────────────────────────┘   │
         │              │                        │                  │
    Input Embedding     │                   Output Embedding        │
    + Pos. Encoding     │                   + Pos. Encoding         │
         │              │                        │                  │
      Inputs            │               Outputs (shifted right)     │
                        └─────────────────────────────────────────┘
```

### Encoder vs Decoder 구성요소 비교 | Component Comparison

| 구성요소 | Encoder | Decoder |
|----------|---------|---------|
| Self-Attention | ✅ (양방향 Bidirectional) | ✅ (단방향 Causal/Masked) |
| Cross-Attention | ❌ | ✅ (Encoder-Decoder Attention) |
| Feed-Forward Network | ✅ | ✅ |
| Add & Layer Norm | ✅ | ✅ |
| 학습 방식 | - | Teacher Forcing |
| 생성 방식 | - | Auto-regressive |

---

## 2. Transformer Encoder

### 2.1 Input Embeddings

**개념 | Concept**

입력 텍스트의 각 단어(토큰)를 연속적인 고차원 벡터로 변환하는 과정.  
*(The process of converting each input word/token into a continuous high-dimensional vector.)*

```
Input Text:  "Professor looking for a Ph.D student"

Vocabulary (예시):
 index 0  →  <PAD>   → [0.0, 0.0, 0.0, 0.0, ...]
 index 13 →  looking → [0.2, 0.1, 0.7, 0.2, ...]  ← 이 벡터를 사용
 index N  →  z       → [0.2, 0.0, 0.7, 0.2, ...]

         ↓ Embedding Lookup
         
 "looking" → [0.2, 0.1, 0.7, 0.2]  (embed_size 차원의 벡터)
```

**핵심 포인트 | Key Points**
- Vocabulary의 각 단어를 `embed_size` 차원의 실수 벡터로 매핑
- **학습 가능한 파라미터** — 훈련을 통해 의미론적으로 유사한 단어들이 가까운 벡터값을 갖게 됨
- Learnable parameters — semantically similar words end up with closer vector representations after training

---

### 2.2 Positional Encoding

**배경 | Background**

Transformer는 RNN과 달리 **병렬 처리**를 하기 때문에, 순서 정보를 별도로 주입해야 함.  
*(Unlike RNNs, Transformers process all tokens in parallel, so positional information must be explicitly injected.)*

```
RNN 방식:           t=1 → t=2 → t=3 → t=4   (순차 처리, sequential)
Transformer 방식:   t=1, t=2, t=3, t=4 동시 처리 + Positional Encoding 추가
```

**Sinusoidal Positional Encoding 수식**

$$PE_{(pos,\ 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos,\ 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

| 기호 | 의미 |
|------|------|
| $pos$ | 문장 내 단어의 위치 (word position in sentence) |
| $i$ | 임베딩 벡터 내 차원 인덱스 (dimension index) |
| $d_{model}$ | 임베딩 벡터의 전체 차원 수 (total embedding dimension) |

**최종 입력 = Input Embedding + Positional Encoding**

```
 Token:    "My"        "bags"      "in"        "bags"
           ┌────┐      ┌────┐      ┌────┐      ┌────┐
Embedding: │ x1 │      │ x2 │      │ x3 │      │ x4 │   ← 의미 벡터
           └────┘      └────┘      └────┘      └────┘
             +           +           +           +
           ┌────┐      ┌────┐      ┌────┐      ┌────┐
Pos.Enc:   │ t1 │      │ t2 │      │ t3 │      │ t4 │   ← 위치 벡터
           └────┘      └────┘      └────┘      └────┘
             ‖           ‖           ‖           ‖
           ┌────┐      ┌────┐      ┌────┐      ┌────┐
Final:     │    │      │    │      │    │      │    │   ← 최종 입력
           └────┘      └────┘      └────┘      └────┘
```

> 💡 "bags" (위치 2)와 "bags" (위치 4)는 **Embedding은 동일**하지만, **Positional Encoding이 달라** 모델이 구분할 수 있음.

---

### 2.3 Self-Attention

**Seq2Seq Attention vs Self-Attention 비교**

```
[기존 Seq2Seq Attention]                    [Transformer Self-Attention]
입력(소스)과 출력(타겟) 사이의 Attention      입력 문장 내 토큰들 사이의 Attention

"어제 식당 갔어 거기 ..."                   "어제  식당  갔어  거기  사람  많더라"
        ↕↕↕                                  ↕ ↕ ↕ ↕ ↕ ↕ (서로를 참조)
"I went to the cafe ..."                   → 각 단어가 같은 문장의 다른 단어들과
(Query ≠ Key = Value)                         관계를 파악 (Query = Key = Value)
```

**Scaled Dot-Product Attention — 6단계**

```
단계 1: Q, K, V 벡터 생성
─────────────────────────────────────────────────────
 Input X → W_Q, W_K, W_V 가중치 행렬과 곱하여
 각 토큰마다 Query(q), Key(k), Value(v) 벡터 생성

단계 2: Attention Score 계산 (Dot Product)
─────────────────────────────────────────────────────
 score = q₁ · k₁ = 112,   q₁ · k₂ = 96

단계 3: Scale
─────────────────────────────────────────────────────
 score / √d_k   →   112/8 = 14,   96/8 = 12
 (Key 벡터 차원의 제곱근으로 나눔 → 그래디언트 안정화)

단계 4: Softmax
─────────────────────────────────────────────────────
 softmax([14, 12])  →  [0.88, 0.12]
 (현재 단어 encoding 시 각 단어가 기여하는 비율)

단계 5: Value 가중합
─────────────────────────────────────────────────────
 0.88 × v₁  +  0.12 × v₂

단계 6: 합산 → Self-Attention 출력 z
─────────────────────────────────────────────────────
 z₁ = Σ (softmax_score_i × v_i)
```

**수식 요약**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Self-Attention의 핵심 특징**

- `Query = Key = Value` → 현재 time step의 모든 hidden states 사용
- 기존 Attention: `Query ≠ Key = Value` (디코더 hidden state가 Query)
- 모든 토큰 쌍 간의 관계를 **O(1) 레이어**에서 직접 계산 → 장거리 의존성(long-range dependency) 포착에 유리

---

### 2.4 Multi-Head Attention

**개념 | Concept**

Self-Attention을 여러 개의 head에서 **병렬로** 수행 → 다양한 표현 공간에서 관계 학습.  
*(Running self-attention in parallel across multiple "heads" to learn diverse representation subspaces.)*

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) × W_O

여기서 각 head:  headᵢ = Attention(Q·W_Qᵢ,  K·W_Kᵢ,  V·W_Vᵢ)
```

**Multi-Head Attention 전체 흐름 (4단계)**

```
1단계: 입력 준비
────────────────────────────────────────────────
 "Thinking Machines" 입력
 → Input Embedding(x1, x2) + Positional Encoding(t1, t2)
 → 최종 입력벡터 x1, x2

2단계: 모든 HEAD 계산 (h=8개)
────────────────────────────────────────────────
 x1, x2
   ├─ HEAD #0: W_Q0, W_K0, W_V0 → Scaled Dot-Product → z₀
   ├─ HEAD #1: W_Q1, W_K1, W_V1 → Scaled Dot-Product → z₁
   ├─ ...
   └─ HEAD #7: W_Q7, W_K7, W_V7 → Scaled Dot-Product → z₇

3단계: 모든 HEAD 연결 (Concatenate)
────────────────────────────────────────────────
 [z₀ | z₁ | z₂ | z₃ | z₄ | z₅ | z₆ | z₇]
  (연결하여 8배 길이의 벡터 생성)

4단계: W_O 행렬곱 → 최종 출력 Z
────────────────────────────────────────────────
 Z = [z₀|z₁|...|z₇] × W_O
 (원래 입력과 동일한 차원으로 압축)
```

**왜 Multi-Head인가? | Why Multiple Heads?**

각 head는 **서로 다른 관계 패턴**을 학습함 (예: 문법적 관계, 의미적 관계, 공지시어 관계 등).  
→ **여러 Self-Attention의 앙상블** 효과.

> 💡 원논문에서는 h=8 heads, d_model=512, 각 head dim = 64

---

### 2.5 Addition & Layer Normalization

#### Residual Connection (Add)

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

```
입력 x ─────────────────────────────────────┐
         │                                   │  (Skip Connection)
         ▼                                   │
   [Self-Attention 또는 FFN]                 │
         │                                   │
         └──────────── + ◀──────────────────┘
                        │
                   Layer Norm
                        │
                    다음 레이어
```

**효과 | Effect**
- 깊은 네트워크에서 **기울기 소실(vanishing gradient)** 문제 완화
- Overfitting 억제, 학습 안정화

#### Layer Normalization vs Batch Normalization

```
Batch Normalization:                 Layer Normalization:
─────────────────────               ─────────────────────
 배치 차원 방향으로 정규화             피처(채널) 차원 방향으로 정규화
 (같은 feature, 다른 샘플들)          (같은 샘플, 다른 feature들)

 [s1: 1 3 6]  → mean=3, std=3       [s1: 1 3 6] → mean=10/3, std=2
 [s2: 2 2 2]  → mean=2, std=0       [s2: 2 2 2] → mean=2,    std=0
 [s3: 0 1 5]  → ...                 [s3: 0 1 5] → ...
```

- **Transformer는 Layer Norm 사용** — 가변 길이 시퀀스에도 안정적으로 동작하기 때문

---

### 2.6 Feed-Forward Networks (FFN)

**구조 | Structure**

각 토큰 위치마다 **독립적으로, 동일하게** 적용되는 2개의 선형 변환.  
*(Applied independently and identically to each position.)*

$$\text{FFN}(x) = \max(0,\ xW_1 + b_1)W_2 + b_2$$

```
입력 x
   │
   ▼
Linear: f₁ = x·W₁ + b₁     (차원 확장: d_model → d_ff, 보통 4배)
   │
   ▼
ReLU: f₂ = max(0, f₁)
   │
   ▼
Linear: f₃ = f₂·W₂ + b₂    (차원 축소: d_ff → d_model)
   │
   ▼
출력 (입력과 동일한 차원)
```

**역할 | Role**
- Self-Attention이 **토큰 간 관계(interaction)**를 학습한다면,
- FFN은 **각 토큰의 표현(representation)**을 비선형적으로 변환
- Position-wise: 각 위치의 토큰을 독립적으로 처리 (병렬 가능)

---

## 3. Transformer Decoder

### 3.1 Causal (Masked) Self-Attention

**개념 | Concept**

Decoder의 Self-Attention에서는 **현재 위치 이전의 토큰만** 참조할 수 있음.  
*(In the Decoder, each position can only attend to earlier positions — not future ones.)*

```
일반 Self-Attention:                  Causal (Masked) Self-Attention:
────────────────────────              ─────────────────────────────────
 A B C D E 모두 서로 참조 가능          A: A만 참조 가능
 (Bidirectional)                       B: A, B 참조 가능
                                       C: A, B, C 참조 가능
                                       D: A, B, C, D 참조 가능
                                       E: A, B, C, D, E 참조 가능
                                       (미래 토큰은 -∞ 마스킹 처리)

Attention Score Matrix:
         A    B    C    D    E
    A  [ ✅  -∞   -∞   -∞   -∞ ]
    B  [ ✅   ✅  -∞   -∞   -∞ ]
    C  [ ✅   ✅   ✅  -∞   -∞ ]
    D  [ ✅   ✅   ✅   ✅  -∞ ]
    E  [ ✅   ✅   ✅   ✅   ✅ ]
        (하삼각 행렬 구조 / Lower-triangular mask)
```

**Auto-regressive 생성 방식**

```
<s> → A → B → C → D → 순서로 한 토큰씩 생성
 │     │    │    │
 A     B    C    D   ← 다음 토큰 예측
```

**Teacher Forcing (훈련 시)**

- 훈련 중에는 이전 예측이 틀려도, **정답 토큰**을 다음 입력으로 사용
- 훈련 안정화 및 수렴 속도 향상

---

### 3.2 Encoder-Decoder Attention (Cross-Attention)

**개념 | Concept**

Decoder가 출력 토큰을 생성할 때, **Encoder의 전체 출력**을 참조하는 메커니즘.  
*(The mechanism by which the Decoder attends to all Encoder outputs when generating each output token.)*

```
Encoder Stack:                          Decoder Stack:
──────────────                          ─────────────
  Encoder 1 ─────────────────────────▶ Decoder 1
  Encoder 2 ─────────────────────────▶ Decoder 2
  Encoder 3 ─────────────────────────▶ Decoder 3
  Encoder 4 ─────────────────────────▶ Decoder 4
  Encoder 5 ─────────────────────────▶ Decoder 5
  Encoder 6 ─────────────────────────▶ Decoder 6
      │                                     │
 "The quick brown fox"           "Der schnelle braune Fuchs"
 (영어 입력)                      (독일어 출력)
```

**Q, K, V의 출처 | Source of Q, K, V**

| 벡터 | 출처 |
|------|------|
| **Query (Q)** | Decoder의 Masked Self-Attention 출력 |
| **Key (K)** | Encoder의 최종 출력 |
| **Value (V)** | Encoder의 최종 출력 |

→ "현재 내가 생성하려는 단어(Q)와 가장 관련 있는 입력 토큰(K,V)은 무엇인가?"

---

### 3.3 Linear & Softmax (출력 생성)

**흐름 | Flow**

```
Decoder Stack 출력 벡터 (d_model 차원)
          │
          ▼
   Linear Layer
   (d_model → vocab_size)   ← "logits"
          │
          ▼
   Softmax
   (vocab_size 차원의 확률 분포)
          │
          ▼
   argmax → 가장 높은 확률의 토큰 선택
   예: index 5 → "am"
```

---

## 📊 전체 정보 흐름 요약 | Full Information Flow Summary

```
[ENCODER]
Input Text → Token IDs → Input Embedding → + Positional Encoding
                                                     │
                              ┌──────────────────────┤
                              │  × N layers:         │
                              │  Self-Attention       │
                              │  → Add & LayerNorm    │
                              │  → Feed-Forward       │
                              │  → Add & LayerNorm    │
                              └──────────────────────┘
                                         │
                                 Encoder Output ──────────┐
                                                          │
[DECODER]                                                 │
Target (shifted right) → Output Embedding                 │
                        + Positional Encoding             │
                                 │                        │
                        ┌────────────────┐                │
                        │  × N layers:   │                │
                        │  Masked Self-  │                │
                        │  Attention     │                │
                        │  → Add & Norm  │                │
                        │  Cross-Attn  ◀─┼────────────────┘
                        │  (Q from dec,  │
                        │   K,V from enc)│
                        │  → Add & Norm  │
                        │  Feed-Forward  │
                        │  → Add & Norm  │
                        └────────────────┘
                                 │
                           Linear + Softmax
                                 │
                          Output Token (예측)
```

---

## 🔑 핵심 개념 정리 | Key Concept Summary

| 개념 | 한국어 설명 | English |
|------|-------------|---------|
| Self-Attention | 같은 시퀀스 내 토큰 간 Attention | Attention within the same sequence |
| Scaled Dot-Product | Q·Kᵀ / √d_k 로 score 계산 | Score = Q·Kᵀ scaled by √d_k |
| Multi-Head | h개 head 병렬 Attention 후 concat | h parallel attention heads, then concat |
| Causal Mask | 미래 토큰 참조 차단 (하삼각 마스크) | Block future tokens with lower-triangular mask |
| Cross-Attention | Decoder Q → Encoder K, V | Decoder queries into Encoder outputs |
| Residual Connection | 입력을 출력에 더함 (Skip) | Add input to sublayer output |
| Layer Norm | 샘플 내 피처 차원으로 정규화 | Normalize across feature dimension per sample |
| Teacher Forcing | 훈련 시 정답 토큰을 다음 입력으로 사용 | Feed ground-truth tokens during training |
| Auto-regressive | 이전 출력을 다음 입력으로 사용 | Use previous output as next input |

