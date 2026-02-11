# PyTorch 디버깅 - 자주 발생하는 에러와 해결법

## 목차
1. [Custom Dataset 구현 시 에러](#1-custom-dataset-구현-시-에러)
2. [Custom Model 구현 시 에러](#2-custom-model-구현-시-에러)
3. [학습 및 평가 시 에러](#3-학습-및-평가-시-에러)
4. [흔한 실수 사례](#4-흔한-실수-사례)

---

## 디버깅 기본 원칙

```
┌─────────────────────────────────────────────────────────────────┐
│                    에러 해결 프로세스                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1️⃣ 에러 메시지 읽기                                           │
│      └─ Python/PyTorch는 어디서, 왜 에러가 발생했는지 알려줌     │
│                                                                 │
│   2️⃣ 키워드 기반 구글 검색                                      │
│      └─ 대부분의 에러는 다른 개발자들이 이미 해결책을 공유함     │
│                                                                 │
│   3️⃣ 코드 비교 및 이해                                          │
│      └─ 해결책을 찾은 후, 자신의 코드와 비교하며 원인 파악       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Custom Dataset 구현 시 에러

### 1-1. `__len__` 메서드 에러

#### 문제 상황

```python
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return 1000  # ❌ 하드코딩된 값
    
    def __getitem__(self, idx):
        return self.data[idx]
```

#### 발생하는 문제

| 상황 | 결과 |
|------|------|
| `__len__` < 실제 데이터 수 | 일부 데이터만 사용됨 |
| `__len__` > 실제 데이터 수 | **IndexError** 발생 |

#### 올바른 코드

```python
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)  # ✅ 실제 데이터 수 반환
    
    def __getitem__(self, idx):
        return self.data[idx]
```

> 📚 검색 키워드: `pytorch custom dataset IndexError`

---

### 1-2. `__getitem__` 메서드 에러

#### 문제 상황

```python
def __getitem__(self, idx):
    return self.X[idx + 1], self.label[idx + 1]  # ❌ 잘못된 인덱스 접근
```

#### 올바른 코드

```python
def __getitem__(self, idx):
    return self.X[idx], self.label[idx]  # ✅ 올바른 인덱스 접근
```

---

### 1-3. 데이터 타입 에러 (RuntimeError)

#### 문제 상황

`nn.Embedding`은 **Long 타입** 텐서를 요구하는데, Float 타입을 입력한 경우

```python
# ❌ Float 타입으로 저장
self.X = torch.tensor(self.seq[:, :-1]).float()

# Embedding layer 통과 시 에러 발생
# RuntimeError: Expected tensor for argument #1 'indices' to have 
# scalar type Long; but got torch.FloatTensor instead
```

#### 올바른 코드

```python
# ✅ Long 타입으로 저장
self.X = torch.tensor(self.seq[:, :-1]).long()

# 또는 __getitem__에서 변환
def __getitem__(self, idx):
    return self.X[idx].long(), self.label[idx].long()
```

#### 레이어별 요구 데이터 타입

| 레이어 | 요구 타입 |
|--------|----------|
| `nn.Embedding` | **Long** (정수) |
| `nn.Linear` | Float |
| `nn.Conv2d` | Float |
| `nn.CrossEntropyLoss` | labels: **Long** |

> 📚 검색 키워드: `nn.Embedding RuntimeError`

---

### 1-4. Dimension Error (CNN 입력)

#### 문제 상황

```python
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
inputs = torch.randn(28, 28)  # ❌ 2D 텐서 (H × W)
out = conv(inputs)
# RuntimeError: Expected 4-dimensional input for 4-dimensional weight
```

#### CNN 입력 차원

```
┌─────────────────────────────────────────────────────────────────┐
│                    CNN 입력 차원 요구사항                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   요구 형식: B × C × H × W  (4D Tensor)                         │
│                                                                 │
│   ┌───────┬─────────┬────────┬────────┐                        │
│   │   B   │    C    │   H    │   W    │                        │
│   │ Batch │ Channel │ Height │ Width  │                        │
│   └───────┴─────────┴────────┴────────┘                        │
│                                                                 │
│   예시: torch.randn(32, 3, 224, 224)                            │
│         └─ 32개 배치, 3채널(RGB), 224×224 이미지                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 올바른 코드

```python
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

# ❌ 2D (H × W)
inputs_2d = torch.randn(28, 28)

# ⚠️ 3D (C × H × W) - 배치 없이 단일 이미지
inputs_3d = torch.randn(1, 28, 28)

# ✅ 4D (B × C × H × W) - 권장
inputs_4d = torch.randn(1, 1, 28, 28)
out = conv(inputs_4d)
```

> 📚 검색 키워드: `Expected 4-dimensional input`

---

## 2. Custom Model 구현 시 에러

### 2-1. Dimension Mismatch Error

#### 문제 상황

Conv layer 출력 크기와 FC layer 입력 크기가 불일치

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),   # [1,28,28] → [16,24,24]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5),  # [16,24,24] → [32,20,20]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),       # [32,20,20] → [32,10,10]
            nn.Conv2d(32, 64, kernel_size=5),  # [32,10,10] → [64,6,6]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),       # [64,6,6] → [64,3,3]
        )
        # ❌ 잘못된 입력 크기: 64*3*3 = 576인데 1600으로 설정
        self.fc_layer = nn.Linear(1600, 10)
```

#### 디버깅 방법

```python
# FC layer 입출력 크기 확인
for name, layer in model.named_modules():
    if isinstance(layer, nn.Linear):
        print(f"Layer {name}: {layer.in_features} -> {layer.out_features}")

# 또는 중간 출력 크기 직접 확인
x = torch.randn(1, 1, 28, 28)
x = model.layer(x)
print(x.shape)  # torch.Size([1, 64, 3, 3])
print(x.view(x.size(0), -1).shape)  # torch.Size([1, 576])
```

#### 올바른 코드

```python
# ✅ 실제 출력 크기에 맞게 수정
self.fc_layer = nn.Linear(64 * 3 * 3, 10)  # 576 → 10
```

> 📚 검색 키워드: `mat1 and mat2 shapes cannot be multiplied`

---

### 2-2. Tensor Manipulation (view/reshape)

#### 문제 상황

Global Average Pooling 후 FC layer 연결 시 차원 불일치

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=5)
        self.gap = nn.AdaptiveAvgPool2d(1)  # 1×1 GAP
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv(x)   # [B, 10, 24, 24]
        x = self.gap(x)    # [B, 10, 1, 1]
        # ❌ view 없이 바로 FC 연결
        x = self.fc(x)     # Error: 4D → 2D 변환 필요
        return x
```

#### 올바른 코드

```python
def forward(self, x):
    x = self.conv(x)           # [B, 10, 24, 24]
    x = self.gap(x)            # [B, 10, 1, 1]
    x = x.view(x.size(0), -1)  # ✅ [B, 10] - Flatten
    x = self.fc(x)             # [B, 10]
    return x
```

#### Flatten 방법들

```python
# 방법 1: view 사용
x = x.view(x.size(0), -1)

# 방법 2: reshape 사용
x = x.reshape(x.size(0), -1)

# 방법 3: flatten 사용
x = x.flatten(start_dim=1)

# 방법 4: nn.Flatten 레이어 사용
self.flatten = nn.Flatten()
x = self.flatten(x)
```

---

## 3. 학습 및 평가 시 에러

### 3-1. CUDA Out of Memory

#### 에러 메시지

```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

#### GPU 메모리 사용처

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU 메모리 사용 요소                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. 미니배치 데이터                                             │
│      └─ batch_size × input_shape                                │
│                                                                 │
│   2. 모델 파라미터                                               │
│      └─ 모든 layer의 weight, bias                               │
│                                                                 │
│   3. 역전파용 중간 결과물                                        │
│      └─ 각 layer의 출력값 (gradient 계산용)                     │
│                                                                 │
│   💡 가장 쉬운 해결: batch_size 줄이기                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 해결 방법

**방법 1: batch_size 감소**
```python
# ❌ 너무 큰 배치
batch_size = 4096

# ✅ 적절한 크기로 감소
batch_size = 256
```

**방법 2: torch.cuda.empty_cache()**
```python
# 사용하지 않는 텐서 삭제
del large_tensor

# GPU 캐시 비우기
torch.cuda.empty_cache()
```

**방법 3: gradient accumulation**
```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (inputs, labels) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss = loss / accumulation_steps  # 손실 스케일링
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**방법 4: mixed precision training**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, labels in dataloader:
    optimizer.zero_grad()
    
    with autocast():  # FP16으로 연산
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

> 📚 검색 키워드: `CUDA out of memory 해결`

---

### 3-2. detach(), cpu(), numpy() 변환

#### 문제 상황

```python
pred = torch.tensor([1., 0., 1.], requires_grad=True).to('cuda')
label = torch.tensor([1., 0., 0.]).to('cuda')

# ❌ 에러 발생
pred_np = pred.numpy()
# RuntimeError: Can't call numpy() on Tensor that requires grad
```

#### 올바른 변환 순서

```
┌─────────────────────────────────────────────────────────────────┐
│                  Tensor → NumPy 변환 순서                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   CUDA Tensor (requires_grad=True)                              │
│         │                                                       │
│         ▼  .detach()  ─── gradient 연결 해제                    │
│         │                                                       │
│   CUDA Tensor (requires_grad=False)                             │
│         │                                                       │
│         ▼  .cpu()  ─── GPU → CPU 이동                           │
│         │                                                       │
│   CPU Tensor                                                    │
│         │                                                       │
│         ▼  .numpy()  ─── Tensor → NumPy 변환                    │
│         │                                                       │
│   NumPy Array                                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 올바른 코드

```python
pred = torch.tensor([1., 0., 1.], requires_grad=True).to('cuda')
label = torch.tensor([1., 0., 0.]).to('cuda')

# ✅ 올바른 변환 순서
pred_np = pred.detach().cpu().numpy()
label_np = label.detach().cpu().numpy()

# sklearn 등 numpy 기반 라이브러리 사용 가능
from sklearn.metrics import accuracy_score
accuracy_score(label_np, pred_np)
```

---

## 4. 흔한 실수 사례

### 4-1. Random Seed 미고정

#### 문제

실험 재현이 불가능 → 하이퍼파라미터 튜닝 결과 신뢰 불가

#### 해결

```python
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

# 학습 시작 전 호출
set_seed(42)
```

---

### 4-2. optimizer.zero_grad() 누락

#### 문제

PyTorch는 기본적으로 **gradient가 누적**됨

```python
# ❌ zero_grad 없이 학습
for inputs, labels in dataloader:
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()  # gradient가 계속 누적됨!
    optimizer.step()
```

#### 해결

```python
# ✅ 매 배치마다 gradient 초기화
for inputs, labels in dataloader:
    optimizer.zero_grad()  # 🔑 필수!
    
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
```

---

### 4-3. model.eval() 누락

#### 문제

평가 시 BatchNorm, Dropout이 학습 모드로 동작

```
┌─────────────────────────────────────────────────────────────────┐
│              train() vs eval() 모드 차이                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Layer        │  model.train()      │  model.eval()           │
│   ─────────────┼─────────────────────┼─────────────────────────│
│   Dropout      │  랜덤하게 뉴런 제거  │  모든 뉴런 사용         │
│   BatchNorm    │  배치 통계 사용      │  학습된 통계 사용       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Dropout 예시

```python
dropout = nn.Dropout(0.5)
input_tensor = torch.randn(5, 10)

# Training Mode - 50% 뉴런이 0으로 설정됨
dropout.train()
output_train = dropout(input_tensor)
print(output_train)
# tensor([[-0.0000,  0.0800, -0.0000, ...]])  # 0이 섞여있음

# Evaluation Mode - 모든 값 유지
dropout.eval()
output_eval = dropout(input_tensor)
print(output_eval)
# tensor([[-0.1234,  0.0400, -0.5678, ...]])  # 0 없음
```

#### 올바른 평가 코드

```python
# ✅ 평가 시 반드시 eval() 호출
model.eval()

with torch.no_grad():  # gradient 계산 비활성화 (메모리 절약)
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        # 평가 로직...

# 다시 학습할 때는 train() 호출
model.train()
```

---

## 에러 해결 체크리스트

```
┌─────────────────────────────────────────────────────────────────┐
│                    PyTorch 에러 체크리스트                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📦 Dataset 에러                                                │
│  □ __len__이 실제 데이터 수를 반환하는가?                       │
│  □ __getitem__ 인덱스가 범위 내인가?                            │
│  □ 데이터 타입이 올바른가? (Embedding→Long, Conv→Float)         │
│                                                                 │
│  🏗️ Model 에러                                                  │
│  □ Conv 출력 크기와 FC 입력 크기가 일치하는가?                  │
│  □ Flatten/view가 필요한 곳에 있는가?                           │
│  □ 입력 텐서 차원이 올바른가? (CNN: 4D, RNN: 3D)                │
│                                                                 │
│  🏃 Training 에러                                               │
│  □ optimizer.zero_grad()를 호출했는가?                          │
│  □ GPU OOM 시 batch_size를 줄였는가?                            │
│  □ Random seed를 고정했는가?                                    │
│                                                                 │
│  📊 Evaluation 에러                                             │
│  □ model.eval()을 호출했는가?                                   │
│  □ torch.no_grad() 컨텍스트를 사용했는가?                       │
│  □ .detach().cpu().numpy() 순서가 올바른가?                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Reference
- [PyTorch 공식 문서](https://pytorch.org/docs/stable/index.html)
- [PyTorch 튜토리얼](https://pytorch.org/tutorials/)
- [CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
