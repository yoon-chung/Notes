# PyTorch DNN 구현 (2) - 학습, 추론, 평가

## 목차
1. [학습 (Training)](#1-학습-training)
2. [추론과 평가 (Inference & Evaluation)](#2-추론과-평가-inference--evaluation)

---

## 1. 학습 (Training)

### 1-1. 손실 함수 (Loss Function)

#### torch.nn 손실 함수

| 손실 함수 | 설명 | 용도 |
|-----------|------|------|
| `nn.NLLLoss` | Negative Log Likelihood | 다중 클래스 분류 (LogSoftmax 출력) |
| `nn.CrossEntropyLoss` | 크로스 엔트로피 | 다중 클래스 분류 (raw logits) |
| `nn.BCELoss` | Binary Cross Entropy | 이진 분류 |
| `nn.MSELoss` | Mean Squared Error | 회귀 |
| `nn.L1Loss` | Mean Absolute Error | 회귀 |

#### 손실 함수 선언

```python
import torch.nn as nn

# 다중 클래스 분류 (모델 출력이 LogSoftmax일 때)
criterion = nn.NLLLoss()

# 다중 클래스 분류 (모델 출력이 raw logits일 때)
criterion = nn.CrossEntropyLoss()
```

> **NLLLoss vs CrossEntropyLoss**
> - `NLLLoss`: 모델 마지막에 `LogSoftmax` 적용 필요
> - `CrossEntropyLoss`: 내부에 Softmax 포함 (raw logits 입력)

---

### 1-2. 최적화 알고리즘 (Optimizer)

#### torch.optim 옵티마이저

| 옵티마이저 | 설명 |
|------------|------|
| `optim.SGD` | 기본 경사 하강법 |
| `optim.Adam` | Adaptive Moment Estimation |
| `optim.AdamW` | Adam with Weight Decay |
| `optim.Adagrad` | Adaptive Gradient |
| `optim.RMSprop` | Root Mean Square Propagation |

#### Optimizer 주요 인자

| 인자 | 설명 |
|------|------|
| `params` | 모델 파라미터 **(필수)** |
| `lr` | 학습률 (learning rate) |
| `weight_decay` | L2 정규화 계수 |
| `momentum` | 모멘텀 (SGD) |
| `betas` | Adam의 β1, β2 |

#### Optimizer 선언

```python
import torch.optim as optim

lr = 0.001

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# SGD with momentum
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Adam with weight decay (L2 regularization)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
```

---

### 1-3. 학습 루프 (Training Loop)

#### 기본 학습 함수

```python
def training(model, dataloader, criterion, optimizer, device):
    model.train()  # 학습 모드 설정
    train_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 1. Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 2. Backward pass
        optimizer.zero_grad()  # 그래디언트 초기화
        loss.backward()        # 역전파
        optimizer.step()       # 가중치 업데이트
        
        # 3. 메트릭 계산
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = train_loss / len(dataloader)
    
    return avg_loss, accuracy
```

#### 검증 함수

```python
def validation(model, dataloader, criterion, device):
    model.eval()  # 평가 모드 설정
    valid_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    avg_loss = valid_loss / len(dataloader)
    
    return avg_loss, accuracy
```

#### Early Stopping이 포함된 전체 학습 루프

```python
def training_loop(model, train_loader, valid_loader, criterion, 
                  optimizer, device, num_epochs, patience):
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 학습
        train_loss, train_acc = training(model, train_loader, 
                                         criterion, optimizer, device)
        
        # 검증
        valid_loss, valid_acc = validation(model, valid_loader, 
                                           criterion, device)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}, Acc: {valid_acc:.4f}')
        
        # Early Stopping 체크
        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')  # 모델 저장
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping!')
                break
    
    return model
```

#### 학습 모드 vs 평가 모드

| 모드 | 함수 | BatchNorm | Dropout |
|------|------|-----------|---------|
| 학습 모드 | `model.train()` | 배치 통계 사용 | 활성화 |
| 평가 모드 | `model.eval()` | 이동 평균 사용 | 비활성화 |

---

### 1-4. 활성화 함수와 가중치 초기화의 중요성

#### 활성화 함수가 없으면?

> **딥러닝 = 선형 변환의 연속**이 되어버림

```
y = W₃(W₂(W₁x)) = (W₃W₂W₁)x = Wx
```

- 활성화 함수 없이는 여러 레이어를 쌓아도 **하나의 선형 변환**과 동일
- **비선형성(non-linearity)**을 도입해야 복잡한 패턴 학습 가능

#### 가중치를 0으로 초기화하면?

> **학습 불가능!**

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial O} \cdot \frac{\partial O}{\partial o_1} \cdot \frac{\partial o_1}{\partial w_1}$$

- 모든 가중치가 0이면 → 출력이 0 → gradient도 0
- Chain rule에서 0이 곱해지면서 **gradient vanishing**

#### 실험 결과 비교

| 실험 | 설정 | Valid Accuracy |
|------|------|----------------|
| exp1 | BatchNorm + Dropout + Activation | **0.9802** |
| exp2 | BatchNorm 제거 | 0.9780 |
| exp3 | Dropout 제거 | 0.9819 |
| exp4 | **Activation 제거** | 0.9147 ⬇️ |
| exp5 | **가중치 0 초기화** | 0.1133 ❌ |

**핵심 인사이트:**
- 활성화 함수 제거 → 성능 급격히 하락 (~6% ↓)
- 가중치 0 초기화 → 학습 실패 (랜덤 수준)

---

## 2. 추론과 평가 (Inference & Evaluation)

### 2-1. 모델 로드 및 추론

#### 저장된 모델 불러오기

```python
# 모델 구조 정의 (학습 때와 동일하게)
model = DNN(hidden_dims=hidden_dims, num_classes=10, ...)

# 저장된 가중치 로드
model.load_state_dict(torch.load('best_model.pt'))
model = model.to(device)
```

#### 추론 (Inference)

```python
model.eval()  # 평가 모드 필수!

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():  # 메모리 절약
    for images, labels in test_dataloader:
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # 예측 클래스 (가장 높은 확률)
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_probs.append(outputs.cpu().numpy())

all_probs = np.vstack(all_probs)
```

### 2-2. 평가 메트릭

#### 주요 분류 메트릭

| 메트릭 | 설명 | 공식 |
|--------|------|------|
| **Accuracy** | 정확도 | (TP + TN) / Total |
| **Precision** | 정밀도 | TP / (TP + FP) |
| **Recall** | 재현율 | TP / (TP + FN) |
| **F1 Score** | 정밀도와 재현율의 조화평균 | 2 × (P × R) / (P + R) |
| **AUC** | ROC 곡선 아래 면적 | - |

#### sklearn으로 메트릭 계산

```python
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score
)

# Precision, Recall, F1
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

# AUC (다중 클래스)
# LogSoftmax 출력이면 exp 적용하여 확률로 변환
all_probs = np.exp(all_probs)  
auc = roc_auc_score(all_labels, all_probs, 
                    average='macro', multi_class='ovr')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'AUC: {auc:.4f}')
```

#### average 파라미터

| 값 | 설명 |
|----|------|
| `'macro'` | 클래스별 메트릭의 단순 평균 |
| `'micro'` | 전체 TP, FP, FN으로 계산 |
| `'weighted'` | 클래스 빈도로 가중 평균 |

---

## 핵심 정리

### 학습 루프 핵심 단계

```python
# 1. Forward pass
outputs = model(inputs)
loss = criterion(outputs, labels)

# 2. Backward pass
optimizer.zero_grad()  # 그래디언트 초기화
loss.backward()        # 역전파
optimizer.step()       # 파라미터 업데이트
```

### model.train() vs model.eval()

```python
model.train()  # 학습 시 (BatchNorm/Dropout 활성화)
model.eval()   # 추론 시 (BatchNorm/Dropout 비활성화)
```

### torch.no_grad()

```python
with torch.no_grad():  # 추론 시 그래디언트 계산 비활성화
    outputs = model(inputs)
# → 메모리 절약, 속도 향상
```

### 모델 저장/로드

```python
# 저장
torch.save(model.state_dict(), 'model.pt')

# 로드
model.load_state_dict(torch.load('model.pt'))
```

---

## Reference
- [Loss Functions - PyTorch 공식 문서](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [torch.optim - PyTorch 공식 문서](https://pytorch.org/docs/stable/optim.html)
- [sklearn.metrics - scikit-learn 공식 문서](https://scikit-learn.org/stable/modules/model_evaluation.html)
