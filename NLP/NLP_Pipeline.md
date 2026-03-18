# NLP Text Classification Pipeline

---

## 1. 파이프라인 개요

### 텍스트 분류(Text Classification) 파이프라인이란?

NLP에서 텍스트 분류는 입력 텍스트를 사전에 정의된 클래스 중 하나로 할당하는 태스크
HuggingFace의 Pre-trained 모델을 Fine-tuning하는 일반적인 파이프라인은 아래 5단계로 구성

```
Step 1. 환경설정
        라이브러리 설치 + 데이터 로드
              │
              ▼
Step 2. 데이터셋 구축
        train/valid split → Tokenizing → PyTorch Dataset
              │
              ▼
Step 3. 모델 & 토크나이저 로드
        HuggingFace Pre-trained 모델 (토크나이저와 동일 체크포인트)
              │
              ▼
Step 4. 학습 (Fine-tuning)
        TrainingArguments + Trainer
              │
              ▼
Step 5. 추론 & 평가
        model.eval() → argmax → accuracy / F1
```

### 이진 분류 태스크 입출력 구조

```
Input  : Text_A (예: 제목) + Text_B (예: 본문)
              │
              ▼
   [ Fine-tuned Classification Model ]
              │
              ▼
Output : 0 or 1  (클래스 레이블)
```

---

## 2. Dataset & Tokenizing

### 2.1 데이터 로드 — `pandas.read_csv`

```python
import pandas as pd

df = pd.read_csv(
    "data/train.csv",
    usecols=["title", "content", "label"],  # 필요한 컬럼만 선택
    dtype={"label": int},                   # 컬럼별 타입 명시
    encoding="utf-8",                       # 한국어: utf-8 or cp949
)
```

**자주 쓰는 파라미터**

| 파라미터 | 역할 | 비고 |
|----------|------|------|
| `sep` | 컬럼 구분자 | 기본값 `,` |
| `header` | 헤더 행 번호 | 기본값 첫 번째 행 |
| `usecols` | 로드할 컬럼 리스트 | 메모리 절약 |
| `dtype` | 컬럼별 데이터 타입 딕셔너리 | 타입 오류 방지 |
| `encoding` | 문자 인코딩 | 한국어: `'utf-8'` 또는 `'cp949'` |

---

### 2.2 PyTorch Dataset Class

PyTorch의 `Dataset` 클래스를 상속받아 커스텀 데이터셋을 만들면, DataLoader와의 연동이 간편해지고 배치 학습·셔플링·멀티프로세싱을 쉽게 구현.  

**반드시 구현해야 할 메서드 3개**

```python
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):

    def __init__(self, encodings, labels):
        """
        초기화: 토크나이징된 입력과 레이블을 저장.
        전처리·토크나이징은 이 단계에서 마치는 것이 일반적.
        """
        self.encodings = encodings  # tokenizer 출력 (input_ids, attention_mask 등)
        self.labels = labels

    def __len__(self):
        """데이터셋 전체 길이 반환 — DataLoader가 인덱싱 범위를 파악할 때 사용"""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 샘플 하나를 딕셔너리 형태로 반환.
        Trainer는 이 딕셔너리를 모델 입력으로 사용.
        """
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
```

| 메서드 | 역할 |
|--------|------|
| `__init__` | 데이터 저장 및 전처리 초기화 |
| `__len__` | 전체 샘플 수 반환 |
| `__getitem__` | 인덱스 기반 단일 샘플 반환 (모델 입력 형태) |

---

### 2.3 Tokenizing — `AutoTokenizer`

```python
from transformers import AutoTokenizer

CHECKPOINT = "klue/roberta-base"  # 사용할 모델 체크포인트
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

# 텍스트 쌍 토크나이징 (예: 제목 + 본문)
encodings = tokenizer(
    list(df["title"]),
    list(df["content"]),
    truncation=True,      # max_length 초과 시 자름
    padding=True,         # 배치 내 길이 통일
    max_length=512,
    return_tensors="pt",  # PyTorch 텐서로 반환
)
```

> **주의**: 토크나이저와 모델은 **반드시 동일한 체크포인트**를 사용해야.  
> 어휘 사전(vocabulary)이 다르면 토큰 ID 불일치로 학습이 불가능.

---

## 3. Model & Trainer

### 3.1 Model — `AutoModelForSequenceClassification`

분류 태스크에는 BERT 계열 Encoder 모델의 `[CLS]` 토큰 출력에 Linear classifier를 붙인 구조가 표준.  

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT,
    num_labels=2,  # 이진 분류: 0 or 1
)
```
---

### 3.2 Trainer

HuggingFace `Trainer`는 학습 루프를 직접 작성하지 않고도 Fine-tuning을 수행할 수 있는 고수준 인터페이스.  

#### TrainingArguments — 학습 하이퍼파라미터 설정

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",

    # 학습 기본 설정
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,

    # 체크포인트 저장
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,           # 최대 저장 체크포인트 수 (오래된 것부터 삭제)

    # 평가
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,  # 학습 종료 시 검증 성능 최고 모델 자동 로드

    # 학습률 스케줄링
    warmup_steps=100,             # 처음 N step 동안 학습률을 0에서 목표값까지 선형 증가
    weight_decay=0.01,

    # 로깅
    logging_dir="./logs",
    logging_steps=100,
)
```

> 💡 **Warmup 직관**: 모델 초기화 직후에는 파라미터가 불안정.  
> 큰 학습률을 바로 적용하면 학습이 발산할 수 있으므로, 초기 N step 동안 학습률을 점진적으로 키움.

**주요 인자 정리**

| 인자 | 설명 |
|------|------|
| `save_total_limit` | 저장할 체크포인트 최대 개수 |
| `warmup_steps` | 학습률 점진 증가 구간 (step 수) |
| `load_best_model_at_end` | 종료 시 검증 최고 성능 체크포인트 자동 로드 |
| `evaluation_strategy` | `"no"` / `"steps"` / `"epoch"` 중 선택 |

#### Trainer — 학습 실행

```python
from transformers import Trainer, EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,    # 평가 함수 연결
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,        # 3번 연속 개선 없으면 조기 종료
            early_stopping_threshold=0.001,   # 이 값 이상 개선돼야 '개선'으로 인정
        )
    ],
)

trainer.train()
```

**Trainer 주요 파라미터**

| 파라미터 | 설명 |
|----------|------|
| `compute_metrics` | 검증 시 사용할 평가 함수 (accuracy, F1 등) |
| `callbacks` | 학습 각 단계마다 실행할 동작 (Early Stopping 등) |
| `optimizers` | 커스텀 optimizer, lr_scheduler 튜플로 전달 |

#### compute_metrics — 평가 함수 정의

```python
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)  # logits → 클래스 인덱스

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="micro")

    return {"accuracy": acc, "f1": f1}
```

#### LR Scheduler — 코사인 스케줄러 예시

```python
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
import torch

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=len(train_dataset) * num_epochs,
)

# Trainer의 optimizers 인자에 튜플로 전달
trainer = Trainer(..., optimizers=(optimizer, scheduler))
```

---

## 4. Inference & Evaluation

### 4.1 Inference

Fine-tuning이 완료된 모델로 테스트 데이터를 추론할 때는 **반드시 평가 모드**로 전환.

```python
import torch
import numpy as np
from torch.utils.data import DataLoader

def run_inference(model, dataset, batch_size, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()           # 평가 모드 전환 (Dropout 비활성화 등)
    predictions = []

    with torch.no_grad():  # gradient 계산 비활성화 → 메모리·속도 최적화
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits  = outputs.logits                     # shape: (batch, num_labels)
            preds   = logits.argmax(dim=-1).cpu().numpy()
            predictions.extend(preds)

    return predictions
```

**`model.eval()` vs `torch.no_grad()` 차이**

| | `model.eval()` | `torch.no_grad()` |
|---|---|---|
| **역할** | 레이어 동작 모드 변경 | gradient 계산 비활성화 |
| **영향받는 것** | Dropout, BatchNorm 등 | 모든 연산의 gradient graph |
| **메모리 절약** | ❌ | ✅ |
| **추론 시 필요** | ✅ 필수 | ✅ 권장 |

> 💡 두 가지는 서로 다른 역할을 하므로, 추론 시에는 **둘 다 사용**하는 것이 표준.

---

### 4.2 Evaluation

```python
from sklearn.metrics import accuracy_score, f1_score, classification_report

acc = accuracy_score(true_labels, predictions)
f1  = f1_score(true_labels, predictions, average="micro")

print(f"Accuracy : {acc:.4f}")
print(f"F1 Score : {f1:.4f}")
print(classification_report(true_labels, predictions))
```

**평가 지표 선택 기준**

| 지표 | 언제 쓰나 |
|------|----------|
| **Accuracy** | 클래스가 균형 잡혀 있을 때 |
| **F1 micro** | 클래스 불균형 데이터, 전체 샘플 기준 성능 필요 시 |
| **F1 macro** | 각 클래스를 동등하게 평가하고 싶을 때 |

---

## 5. 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Text Classification** | 텍스트를 사전 정의 클래스로 분류하는 지도학습 태스크 |
| **Fine-tuning** | Pre-trained 모델을 특정 태스크 데이터로 추가 학습 |
| **AutoTokenizer** | 체크포인트에 맞는 토크나이저 자동 로드 |
| **AutoModelForSequenceClassification** | `[CLS]` 출력 + Linear classifier 구조의 분류 모델 |
| **PyTorch Dataset** | `__init__` / `__len__` / `__getitem__` 구현 필수 |
| **Trainer** | HuggingFace 학습 루프 추상화 클래스 |
| **Warmup** | 초기 불안정 구간에서 학습률을 점진적으로 증가시키는 기법 |
| **Early Stopping** | 검증 성능이 patience 횟수 이상 개선되지 않으면 조기 종료 |
| **model.eval()** | Dropout·BatchNorm을 추론 모드로 전환 |
| **torch.no_grad()** | gradient 계산 비활성화 → 추론 속도·메모리 최적화 |
| **F1 (micro)** | 클래스 불균형 상황에서 전체 TP/FP/FN 기준으로 계산 |

