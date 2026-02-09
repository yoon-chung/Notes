# PyTorch DNN 구현 (1) - Dataset, DataLoader, Custom Model

## 목차
1. [데이터](#1-데이터)
2. [모델](#2-모델)

---

## 1. 데이터

### 1-1. torchvision으로 Dataset 불러오기

#### torchvision.datasets

PyTorch의 `torchvision.datasets` 모듈에서 다양한 비전 데이터셋을 간편하게 불러올 수 있음

| 데이터셋 | 설명 |
|----------|------|
| `torchvision.datasets.MNIST` | 손글씨 숫자 (0~9), 28×28 |
| `torchvision.datasets.CIFAR10` | 10개 클래스 이미지 |
| `torchvision.datasets.CIFAR100` | 100개 클래스 이미지 |
| `torchvision.datasets.ImageNet` | 대규모 이미지 데이터셋 |

#### MNIST 데이터셋 불러오기

```python
import torchvision
import torchvision.transforms as T

# Transform 정의
mnist_transform = T.Compose([
    T.ToTensor(),  # 텐서로 변환 (0~255 → 0~1)
])

# Dataset 다운로드 및 불러오기
download_root = './MNIST_DATASET'

train_dataset = torchvision.datasets.MNIST(
    download_root, 
    transform=mnist_transform, 
    train=True,      # 학습 데이터
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    download_root, 
    transform=mnist_transform, 
    train=False,     # 테스트 데이터
    download=True
)
```

#### 데이터 구조 확인

```python
for image, label in train_dataset:
    print(image.shape, label)  # torch.Size([1, 28, 28]) 5
    break
# shape: [C, H, W] = [채널, 높이, 너비]
```

#### Train/Validation 분리

```python
from torch.utils.data import random_split

total_size = len(train_dataset)
train_num = int(total_size * 0.8)  # 80%
valid_num = int(total_size * 0.2)  # 20%

train_dataset, valid_dataset = random_split(
    train_dataset, 
    [train_num, valid_num]
)
```

---

### 1-2. DataLoader 정의

`DataLoader`는 Dataset을 미니 배치(mini-batch)로 묶어주는 역할

#### DataLoader 주요 인자

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `dataset` | 데이터셋 객체 **(필수)** | - |
| `batch_size` | 미니 배치 크기 | 1 |
| `shuffle` | 에폭마다 데이터 순서 섞기 | False |
| `num_workers` | 데이터 로딩 프로세스 수 | 0 |
| `drop_last` | 마지막 불완전 배치 버리기 | False |
| `pin_memory` | GPU 전송 속도 향상 (CUDA) | False |

#### DataLoader 선언

```python
from torch.utils.data import DataLoader

batch_size = 32

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True    # 학습용은 shuffle=True
)

valid_dataloader = DataLoader(
    valid_dataset, 
    batch_size=batch_size, 
    shuffle=False   # 검증/테스트는 shuffle=False
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False
)
```

#### DataLoader 사용

```python
for images, labels in train_dataloader:
    print(images.shape, labels.shape)
    # torch.Size([32, 1, 28, 28]) torch.Size([32])
    # [배치, 채널, 높이, 너비]
    break
```

---

## 2. 모델

### 2-1. nn.Module로 Custom Model 정의

#### DNN 모델 구조

MNIST (28×28 이미지) → 1차원으로 펼침 (784) → FC Layers → 10개 클래스

```
Input (784) → FC1 (512) → FC2 (256) → FC3 (128) → Output (10)
```

#### 기본 DNN 모델

```python
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()  # 필수!
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(28 * 28, hidden_dim * 4)  # 784 → 512
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)  # 512 → 256
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)  # 256 → 128
        self.classifier = nn.Linear(hidden_dim, num_classes)  # 128 → 10
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # [B, 1, 28, 28] → [B, 784]
        x = x.view(x.shape[0], -1)
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.classifier(x)
        
        return x
```

#### 유연한 DNN 모델 (ModuleList 활용)

```python
class DNN(nn.Module):
    def __init__(self, hidden_dims, num_classes, dropout_ratio=0.2,
                 apply_batchnorm=True, apply_dropout=True, apply_activation=True):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()
        
        # 레이어 동적 생성
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            
            if apply_batchnorm:
                self.layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            
            if apply_activation:
                self.layers.append(nn.ReLU())
            
            if apply_dropout:
                self.layers.append(nn.Dropout(dropout_ratio))
        
        # 분류기
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.classifier(x)
        return x
```

#### 모델 생성 및 테스트

```python
hidden_dims = [784, 512, 256, 128]  # 입력 → 히든 레이어들

model = DNN(
    hidden_dims=hidden_dims,
    num_classes=10,
    dropout_ratio=0.2,
    apply_batchnorm=True,
    apply_dropout=True,
    apply_activation=True
)

# 테스트: 랜덤 입력으로 연산 확인
output = model(torch.randn(32, 1, 28, 28))
print(output.shape)  # torch.Size([32, 10])
```

#### super().__init__()의 중요성

> ⚠️ `super().__init__()`을 호출하지 않으면 에러 발생!

```python
class DNN(nn.Module):
    def __init__(self):
        # super().__init__()  ← 이걸 안 하면
        self.fc1 = nn.Linear(784, 512)  # ❌ 에러!
```

`nn.Module`의 `__init__`에서 파라미터, 버퍼, 서브 모듈 등을 자동 등록/추적하므로 반드시 호출해야 함

---

### 2-2. 가중치 초기화

#### nn.init 주요 함수

| 함수 | 설명 |
|------|------|
| `nn.init.zeros_(tensor)` | 모든 값을 0으로 |
| `nn.init.ones_(tensor)` | 모든 값을 1로 |
| `nn.init.constant_(tensor, val)` | 상수로 초기화 |
| `nn.init.uniform_(tensor, a, b)` | 균등분포 [a, b] |
| `nn.init.normal_(tensor, mean, std)` | 정규분포 |
| `nn.init.xavier_normal_(tensor)` | Xavier 초기화 |
| `nn.init.kaiming_normal_(tensor)` | Kaiming(He) 초기화 |

#### Xavier vs Kaiming 초기화

| 초기화 | 표준편차 | 권장 활성화 함수 |
|--------|----------|------------------|
| Xavier | $\sqrt{\frac{2}{n_{in} + n_{out}}}$ | Sigmoid, Tanh |
| Kaiming | $\sqrt{\frac{2}{n_{in}}}$ | ReLU |

#### 가중치 초기화 함수 구현

```python
def weight_initialization(model, method):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            # 가중치 초기화
            if method == 'gaussian':
                nn.init.normal_(m.weight)
            elif method == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(m.weight)
            elif method == 'zeros':
                nn.init.zeros_(m.weight)
            
            # 편향은 0으로 초기화
            nn.init.zeros_(m.bias)
    
    return model
```

#### 모델 내부에 초기화 메서드 추가

```python
class DNN(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... 레이어 정의
    
    def forward(self, x):
        # ... forward pass
    
    def weight_initialization(self, method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if method == 'gaussian':
                    nn.init.normal_(m.weight)
                elif method == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  # numel(): 텐서의 원소개수를 반환하는 함수
```

#### 사용 예시

```python
model = DNN(hidden_dims=[784, 512, 256, 128], num_classes=10, ...)

# 가중치 초기화
model.weight_initialization('kaiming')

# 파라미터 수 확인
print(f'Trainable parameters: {model.count_parameters():,}')
# Trainable parameters: 569,226
```

---

## 핵심 정리

### Dataset & DataLoader 흐름

```
torchvision.datasets → Dataset 객체
        ↓
random_split() → train/valid 분리
        ↓
DataLoader() → 미니 배치 생성
        ↓
for images, labels in dataloader: → 학습 루프
```

### Custom Model 구조

```python
class MyModel(nn.Module):
    def __init__(self, ...):
        super().__init__()      # 1. 부모 클래스 초기화
        self.layer = nn.Linear(...)  # 2. 레이어 정의
    
    def forward(self, x):       # 3. forward pass 정의
        return self.layer(x)
```

### 가중치 초기화 선택 가이드

| 상황 | 추천 초기화 |
|------|-------------|
| ReLU 사용 | **Kaiming** |
| Sigmoid/Tanh 사용 | **Xavier** |
| 일반적인 경우 | Kaiming 또는 Xavier |

---

## Reference
- [torchvision.datasets - PyTorch 공식 문서](https://pytorch.org/vision/stable/datasets.html)
- [torch.nn.init - PyTorch 공식 문서](https://pytorch.org/docs/stable/nn.init.html)
- [가중치 초기화](https://freshrimpsushi.github.io/posts/weights-initialization-in-pytorch/)
