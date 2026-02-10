# PyTorch RNN 구현 - Vanilla RNN, LSTM, GRU, Bidirectional

## 목차
1. [Vanilla RNN](#1-vanilla-rnn)
2. [LSTM](#2-lstm)
3. [GRU](#3-gru)
4. [Bidirectional RNN/LSTM/GRU](#4-bidirectional-rnnlstmgru)

---

## 1. Vanilla RNN

### 1-1. RNN이란?

**Recurrent Neural Network**: 입력과 출력을 **시퀀스(sequence) 단위**로 처리하는 모델

![RNN Unrolled](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)

*출처: colah's blog - Understanding LSTM Networks*

#### 핵심 특징

- **시간 흐름에 따라 정보 공유**: 이전 단계의 정보가 현재 단계에 영향
- **Hidden State**: 이전 시점의 은닉 상태 + 현재 입력 → 다음 은닉 상태 계산
- 초기 hidden state: zero-vector 또는 random 초기화

#### RNN 수식

$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$$
$$y_t = W_{hy}h_t + b_y$$

| 기호 | 의미 |
|------|------|
| $h_t$ | t 시점의 hidden state |
| $x_t$ | t 시점의 입력 |
| $h_{t-1}$ | 이전 시점의 hidden state |

---

### 1-2. nn.RNN 사용법

#### 주요 파라미터

| 파라미터 | 설명 |
|----------|------|
| `input_size` | 입력 차원 (embedding dim) |
| `hidden_size` | hidden state 차원 |
| `num_layers` | RNN 레이어 수 (기본값: 1) |
| `batch_first` | True면 입력 shape이 [batch, seq, feature] |
| `bidirectional` | 양방향 여부 |

#### RNN Output

```python
output, h_n = self.rnn(x)
```

| 반환값 | Shape | 설명 |
|--------|-------|------|
| `output` | [batch, seq_len, hidden_size] | 모든 time step의 hidden state |
| `h_n` | [num_layers, batch, hidden_size] | 마지막 time step의 hidden state |

---

### 1-3. RNN 모델 구현 (Next Word Prediction)

```python
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        
        # Embedding
        x = self.embedding(x)  # [batch, seq_len, embedding_dim]
        
        # RNN
        output, h_n = self.rnn(x)  # output: [batch, seq_len, hidden_size]
        
        # 마지막 time step만 사용
        last_hidden = output[:, -1, :]  # [batch, hidden_size]
        
        # Classification
        out = self.fc(last_hidden)  # [batch, vocab_size]
        
        return out
```

> **왜 마지막 time step만 사용?**
> - RNN은 이전 정보를 현재로 전달하므로, 마지막 hidden state가 **모든 이전 정보를 포함**
> - Next word prediction에서는 전체 문맥을 반영한 마지막 상태로 예측

---

## 2. LSTM

### 2-1. LSTM이란?

**Long Short-Term Memory**: RNN의 **장기 의존성 문제**를 해결한 모델

![LSTM Chain](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

*출처: colah's blog - Understanding LSTM Networks*

#### RNN의 장기 의존성 문제

> 시퀀스가 길어질수록 **앞부분의 정보가 뒷부분 예측에 영향을 미치기 어려움**
> (Vanishing Gradient 문제)

#### LSTM의 해결책: Gate 메커니즘

![LSTM Gates](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png)

*Forget Gate - 과거 정보 중 얼마나 버릴지 결정*

| Gate | 역할 |
|------|------|
| **Forget Gate** | 과거 정보 중 **얼마나 버릴지** 결정 |
| **Input Gate** | 새로운 정보를 **얼마나 기억할지** 결정 |
| **Output Gate** | 다음 시점으로 **얼마나 전달할지** 결정 |

#### Cell State

![Cell State](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-C-line.png)

*Cell State - 장기 기억을 저장하는 별도의 경로*

- LSTM의 핵심: **Cell State** ($C_t$)
- 장기 기억을 저장하는 별도의 경로
- Gate를 통해 정보를 추가/삭제

---

### 2-2. nn.LSTM 사용법

#### LSTM Output

```python
output, (h_n, c_n) = self.lstm(x)
```

| 반환값 | Shape | 설명 |
|--------|-------|------|
| `output` | [batch, seq_len, hidden_size] | 모든 time step의 hidden state |
| `h_n` | [num_layers, batch, hidden_size] | 마지막 hidden state |
| `c_n` | [num_layers, batch, hidden_size] | 마지막 **cell state** |

> RNN과 달리 **cell state** (`c_n`)도 반환!

---

### 2-3. LSTM 모델 구현

```python
class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        
        x = self.embedding(x)  # [batch, seq_len, embedding_dim]
        
        output, (h_n, c_n) = self.lstm(x)
        
        # 마지막 time step 사용
        last_hidden = output[:, -1, :]  # [batch, hidden_size]
        
        out = self.fc(last_hidden)  # [batch, vocab_size]
        
        return out
```

---

## 3. GRU

### 3-1. GRU란?

**Gated Recurrent Unit**: LSTM을 **간소화**한 모델

![GRU](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png)

*출처: colah's blog - Understanding LSTM Networks*

#### LSTM vs GRU

| 비교 | LSTM | GRU |
|------|------|-----|
| Gate 수 | 3개 (Forget, Input, Output) | 2개 (Reset, Update) |
| Cell State | 있음 | **없음** |
| 파라미터 수 | 더 많음 | **더 적음** |
| 연산 비용 | 더 높음 | **더 낮음** |
| 성능 | 비슷 | 비슷 |

#### GRU Gate

| Gate | 역할 |
|------|------|
| **Reset Gate** | 이전 정보 중 **무엇을 무시할지** 결정 |
| **Update Gate** | 새 정보를 **얼마나 반영할지** 결정 |

---

### 3-2. nn.GRU 사용법

#### GRU Output

```python
output, h_n = self.gru(x)
```

| 반환값 | Shape | 설명 |
|--------|-------|------|
| `output` | [batch, seq_len, hidden_size] | 모든 time step의 hidden state |
| `h_n` | [num_layers, batch, hidden_size] | 마지막 hidden state |

> LSTM과 달리 **cell state 없음** (RNN과 동일한 output 구조)

---

### 3-3. GRU 모델 구현

```python
class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        
        x = self.embedding(x)  # [batch, seq_len, embedding_dim]
        
        output, h_n = self.gru(x)
        
        # 마지막 time step 사용
        last_hidden = output[:, -1, :]  # [batch, hidden_size]
        
        out = self.fc(last_hidden)  # [batch, vocab_size]
        
        return out
```

---

## 4. Bidirectional RNN/LSTM/GRU

### 4-1. Bidirectional이란?

**양방향 RNN**: **과거 + 미래** 정보를 모두 활용

![Bidirectional RNN](https://d2l.ai/_images/birnn.svg)

*출처: Dive into Deep Learning (d2l.ai)*

#### 왜 양방향?

```
1. I'm ____.
   → sad, happy, hungry 등 다양한 단어 가능

2. I'm ____ hungry.
   → very, not 등 더 제한된 단어

3. I'm ____ hungry, so I can eat more.
   → "not"이 가장 적절! (뒤 문맥으로 유추)
```

> **뒤의 문맥**도 예측에 중요한 정보!

---

### 4-2. Bidirectional 구현

#### 핵심: `bidirectional=True`

```python
nn.RNN(input_size, hidden_size, bidirectional=True)
nn.LSTM(input_size, hidden_size, bidirectional=True)
nn.GRU(input_size, hidden_size, bidirectional=True)
```

#### Output Shape 변화

| 구분 | 단방향 | 양방향 |
|------|--------|--------|
| output | [batch, seq, hidden] | [batch, seq, **hidden×2**] |
| h_n | [layers, batch, hidden] | [**layers×2**, batch, hidden] |

> **hidden size가 2배**가 됨! (forward + backward)

---

### 4-3. Bidirectional 모델 구현

```python
class Bidirectional(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, model_type):
        super(Bidirectional, self).__init__()
        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 모델 타입에 따라 선택
        if model_type == 'bi_RNN':
            self.model = nn.RNN(embedding_dim, hidden_size, 
                               batch_first=True, bidirectional=True)
        elif model_type == 'bi_LSTM':
            self.model = nn.LSTM(embedding_dim, hidden_size, 
                                batch_first=True, bidirectional=True)
        elif model_type == 'bi_GRU':
            self.model = nn.GRU(embedding_dim, hidden_size, 
                               batch_first=True, bidirectional=True)
        
        # 양방향이므로 hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        
        if self.model_type == 'bi_LSTM':
            output, (h_n, c_n) = self.model(x)
        else:
            output, h_n = self.model(x)
        
        # 마지막 time step (양방향이므로 hidden_size * 2)
        last_hidden = output[:, -1, :]
        
        out = self.fc(last_hidden)
        
        return out
```

---

## 모델 비교 정리

### RNN vs LSTM vs GRU

| 모델 | 장기 의존성 | Gate 수 | Cell State | 파라미터 | 속도 |
|------|------------|---------|------------|----------|------|
| **RNN** | ❌ 취약 | 0 | ❌ | 적음 | 빠름 |
| **LSTM** | ✅ 해결 | 3 | ✅ | 많음 | 느림 |
| **GRU** | ✅ 해결 | 2 | ❌ | 중간 | 중간 |

### Output 구조 비교

| 모델 | 반환값 |
|------|--------|
| `nn.RNN` | output, h_n |
| `nn.LSTM` | output, **(h_n, c_n)** |
| `nn.GRU` | output, h_n |

### Next Word Prediction 성능 비교

| 모델 | Test Accuracy |
|------|---------------|
| RNN | 12.75% |
| LSTM | 13.50% |
| GRU | 13.85% |
| bi_RNN | 12.27% |
| bi_LSTM | 13.12% |
| bi_GRU | 13.80% |

> 이 데이터셋에서는 GRU가 가장 좋은 성능을 보임

---

## 핵심 코드 패턴

### 공통 구조

```python
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        
        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 2. RNN/LSTM/GRU
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        
        # 3. FC Layer
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)           # [B, seq, emb]
        output, h_n = self.rnn(x)       # output: [B, seq, hidden]
        last = output[:, -1, :]         # [B, hidden] ← 마지막 time step
        out = self.fc(last)             # [B, vocab]
        return out
```

### Bidirectional 수정 포인트

```python
# 1. bidirectional=True 추가
self.rnn = nn.RNN(..., bidirectional=True)

# 2. FC layer input size를 2배로
self.fc = nn.Linear(hidden_size * 2, vocab_size)
```

### 손실 함수 (패딩 무시)

```python
criterion = nn.CrossEntropyLoss(ignore_index=0)  # <pad> 토큰 무시
```

---

## Reference
- [colah's blog - Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - 이미지 출처
- [Dive into Deep Learning (d2l.ai)](https://d2l.ai/) - Bidirectional RNN 이미지
- [PyTorch RNN 공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [PyTorch LSTM 공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [PyTorch GRU 공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [LSTM 논문](https://dl.acm.org/doi/10.1162/neco.1997.9.8.1735)
- [GRU 논문](https://arxiv.org/abs/1406.1078)
