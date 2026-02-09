# PyTorch DNN 구현 (3) - Custom Dataset & Next Word Prediction

## 목차
1. [Custom Dataset 구축](#1-custom-dataset-구축)
2. [Next Word Prediction 모델](#2-next-word-prediction-모델)

---

## 1. Custom Dataset 구축

### 1-1. 자연어 데이터 전처리

#### Next Word Prediction이란?

주어진 텍스트의 **다음 단어를 예측**하는 태스크

| Input | Label |
|-------|-------|
| 나는 | 학교를 |
| 나는 학교를 | 가서 |
| 나는 학교를 가서 | 밥을 |
| 나는 학교를 가서 밥을 | 먹었다 |

#### 텍스트 클리닝

```python
import re

def cleaning_text(text):
    # 특수문자 제거 (영문, 숫자, 기본 문장부호만 유지)
    cleaned_text = re.sub(r"[^a-zA-Z0-9.,@#!\s']+", "", text)
    # No-break space 처리
    cleaned_text = cleaned_text.replace(u'\xa0', u' ')
    cleaned_text = cleaned_text.replace('\u200a', ' ')
    return cleaned_text

# 전체 데이터에 적용
cleaned_data = list(map(cleaning_text, data))
```

> **No-break space**: 웹 크롤링 데이터에서 자주 발생하는 특수 공백 문자 (`\xa0`, `\u200a`)

---

### 1-2. Tokenizer

텍스트를 **단어 단위로 분리**하는 도구

#### torchtext tokenizer 사용

```python
from torchtext.data import get_tokenizer

tokenizer = get_tokenizer("basic_english")

text = "I love PyTorch"
tokens = tokenizer(text)
# ['i', 'love', 'pytorch']
```

#### Tokenizer의 역할

1. **토큰 분리**: 문장 → 단어/하위 단위로 분리
2. **숫자 매핑**: 단어 → 고유 ID로 변환

---

### 1-3. Vocabulary (단어 사전) 구축

#### build_vocab_from_iterator

```python
import torchtext

# 단어 사전 생성
vocab = torchtext.vocab.build_vocab_from_iterator(
    map(tokenizer, cleaned_data),
    min_freq=1  # 최소 빈도 수
)

# 패딩 토큰 추가 (인덱스 0)
vocab.insert_token('<pad>', 0)
```

#### Vocab 주요 메서드

| 메서드 | 설명 | 예시 |
|--------|------|------|
| `get_itos()` | ID → 문자열 | `['<pad>', 'to', 'the', ...]` |
| `get_stoi()` | 문자열 → ID | `{'<pad>': 0, 'to': 1, ...}` |
| `lookup_indices(tokens)` | 토큰 리스트 → ID 리스트 | `[3, 273, 66, 1]` |

```python
# ID → Token
id2token = vocab.get_itos()
print(id2token[:5])  # ['<pad>', 'to', 'the', 'a', 'of']

# Token → ID
token2id = vocab.get_stoi()
print(token2id['the'])  # 2

# 문장을 ID 시퀀스로 변환
text = "A Beginners Guide to Word"
ids = vocab.lookup_indices(tokenizer(text))
# [3, 273, 66, 1, 467]
```

---

### 1-4. 시퀀스 생성 및 Padding

#### Next Word Prediction용 시퀀스 생성

```python
seq = []
for text in cleaned_data:
    token_ids = vocab.lookup_indices(tokenizer(text))
    # 점진적으로 시퀀스 생성
    for j in range(1, len(token_ids)):
        sequence = token_ids[:j+1]
        seq.append(sequence)

# 예시: "a beginners guide to" → [3, 273, 66, 1]
# seq[0] = [3, 273]           → input: [3], label: 273
# seq[1] = [3, 273, 66]       → input: [3, 273], label: 66
# seq[2] = [3, 273, 66, 1]    → input: [3, 273, 66], label: 1
```

#### Zero Padding (앞쪽 패딩)

입력 길이를 동일하게 맞추기 위해 **앞쪽에 0을 채움**

```python
def pre_zeropadding(seq, max_len):
    result = []
    for s in seq:
        if len(s) >= max_len:
            result.append(s[:max_len])
        else:
            # 앞쪽에 0 패딩
            padded = [0] * (max_len - len(s)) + s
            result.append(padded)
    return np.array(result)

max_len = 24
padded_seq = pre_zeropadding(seq, max_len)

# [3, 273] → [0, 0, 0, ..., 0, 3, 273]  (길이 24)
```

#### Input / Label 분리

```python
input_x = padded_seq[:, :-1]  # 마지막 토큰 제외
label = padded_seq[:, -1]     # 마지막 토큰만

# input: [0, 0, ..., 0, 3]    (길이 23)
# label: 273
```

---

### 1-5. Custom Dataset 클래스 구현

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, vocab, tokenizer, max_len):
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # 시퀀스 생성
        seq = self.make_sequence(data, vocab, tokenizer)
        # 패딩 적용
        seq = self.pre_zeropadding(seq, max_len)
        
        # Input과 Label 분리
        self.X = torch.tensor(seq[:, :-1])
        self.label = torch.tensor(seq[:, -1])
    
    def make_sequence(self, data, vocab, tokenizer):
        seq = []
        for text in data:
            token_ids = vocab.lookup_indices(tokenizer(text))
            for j in range(1, len(token_ids)):
                seq.append(token_ids[:j+1])
        return seq
    
    def pre_zeropadding(self, seq, max_len):
        return np.array([
            s[:max_len] if len(s) >= max_len 
            else [0] * (max_len - len(s)) + s 
            for s in seq
        ])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.label[idx]
```

#### Dataset 사용

```python
from sklearn.model_selection import train_test_split

# 데이터 분할 (8:1:1)
train, test = train_test_split(data, test_size=0.2, random_state=42)
val, test = train_test_split(test, test_size=0.5, random_state=42)

# Dataset 생성
train_dataset = CustomDataset(train, vocab, tokenizer, max_len=20)
valid_dataset = CustomDataset(val, vocab, tokenizer, max_len=20)
test_dataset = CustomDataset(test, vocab, tokenizer, max_len=20)

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

---

## 2. Next Word Prediction 모델

### 2-1. nn.Embedding

**단어(정수 ID)를 고정 길이 벡터로 변환**하는 레이어

#### Embedding이란?

- 텍스트/범주형 데이터 → 저차원 실수 벡터로 매핑
- 비슷한 의미의 단어는 비슷한 벡터로 학습됨

#### nn.Embedding 파라미터

| 파라미터 | 설명 |
|----------|------|
| `num_embeddings` | 임베딩할 단어 수 (vocab size) |
| `embedding_dim` | 임베딩 벡터 차원 |
| `padding_idx` | 패딩 인덱스 (gradient 계산 제외) |

```python
import torch.nn as nn

# vocab_size=10000, embedding_dim=512
embedding = nn.Embedding(
    num_embeddings=10000,
    embedding_dim=512,
    padding_idx=0  # <pad> 토큰은 학습에서 제외
)

# 입력: [batch, seq_len] (정수 ID)
# 출력: [batch, seq_len, embedding_dim]
```

---

### 2-2. Next Word Prediction DNN 모델

```python
class NextWordPredictionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dims, hidden_dims, 
                 num_classes, dropout_ratio):
        super().__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dims, 
            padding_idx=0
        )
        
        # FC Layers
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_ratio))
        
        # Classifier
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        # x: [batch, seq_len]
        
        # Embedding
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [batch, seq_len * embed_dim]
        
        # FC Layers
        for layer in self.layers:
            x = layer(x)
        
        # Classification
        x = self.classifier(x)
        x = self.log_softmax(x)
        
        return x
```

### 2-3. 모델 학습

```python
# 하이퍼파라미터
lr = 1e-3
vocab_size = len(vocab.get_stoi())
embedding_dims = 512
hidden_dims = [embedding_dims, embedding_dims*4, embedding_dims*2, embedding_dims]

# 모델 생성
model = NextWordPredictionModel(
    vocab_size=vocab_size,
    embedding_dims=embedding_dims,
    hidden_dims=hidden_dims,
    num_classes=vocab_size,  # 예측할 단어 수 = vocab size
    dropout_ratio=0.2
).to(device)

# 손실 함수 (패딩 인덱스 무시)
criterion = nn.NLLLoss(ignore_index=0)

# 옵티마이저
optimizer = optim.Adam(model.parameters(), lr=lr)
```

> **ignore_index=0**: `<pad>` 토큰에 대한 손실은 계산하지 않음

### 2-4. 추론

```python
model.load_state_dict(torch.load('model_next.pt'))
model.eval()

total_preds = []
total_labels = []

with torch.no_grad():
    for texts, labels in test_loader:
        texts = texts.to(device)
        
        outputs = model(texts)
        _, predicted = torch.max(outputs.data, 1)
        
        total_preds.extend(predicted.cpu().tolist())
        total_labels.extend(labels.tolist())

# Accuracy 계산
accuracy = sum(p == l for p, l in zip(total_preds, total_labels)) / len(total_labels)
```

---

## 핵심 정리

### 자연어 전처리 파이프라인

```
Raw Text
    ↓ cleaning_text()
Cleaned Text
    ↓ tokenizer()
Tokens ['i', 'love', 'pytorch']
    ↓ vocab.lookup_indices()
Token IDs [1, 234, 567]
    ↓ make_sequence()
Sequences [[1, 234], [1, 234, 567]]
    ↓ pre_zeropadding()
Padded [[0, 0, 1, 234], [0, 1, 234, 567]]
    ↓ split input/label
Input: [0, 0, 1], Label: 234
```

### Custom Dataset 필수 메서드

```python
class CustomDataset(Dataset):
    def __init__(self, ...):     # 데이터 로드 & 전처리
        pass
    
    def __len__(self):           # 데이터셋 크기
        return len(self.data)
    
    def __getitem__(self, idx):  # 단일 샘플 반환
        return self.X[idx], self.label[idx]
```

### Embedding Layer

```python
nn.Embedding(
    num_embeddings=vocab_size,  # 단어 수
    embedding_dim=512,          # 벡터 차원
    padding_idx=0               # 패딩 토큰 (학습 제외)
)
# 입력: [batch, seq_len] (정수)
# 출력: [batch, seq_len, embed_dim] (실수 벡터)
```

### NLLLoss with ignore_index

```python
criterion = nn.NLLLoss(ignore_index=0)
# → 패딩 토큰(0)에 대한 손실은 무시
```

---

## Reference
- [torchtext.vocab - PyTorch 공식 문서](https://pytorch.org/text/stable/vocab.html)
- [nn.Embedding - PyTorch 공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [Custom Dataset 구축 - PyTorch 튜토리얼](https://tutorials.pytorch.kr/beginner/basics/data_tutorial.html)
