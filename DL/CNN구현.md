# PyTorch CNN 구현 - Custom CNN, VGG, GoogLeNet, ResNet

## 목차
1. [기본 CNN 모델](#1-기본-cnn-모델)
2. [VGG 모델](#2-vgg-모델)
3. [GoogLeNet 모델](#3-googlenet-모델)
4. [ResNet 모델](#4-resnet-모델)

---

## 1. 기본 CNN 모델

### 1-1. Convolution 연산

#### Convolution이란?

이미지의 픽셀과 필터(커널)를 **원소별 곱셈 후 합산**하는 연산

![convolution](https://gaussian37.github.io/assets/img/dl/concept/conv/1.gif)

#### Output Size 계산 공식

$$O = \lfloor \frac{I + 2P - K}{S} \rfloor + 1$$

| 기호 | 의미 |
|------|------|
| $O$ | Output size |
| $I$ | Input size |
| $P$ | Padding size |
| $K$ | Kernel(Filter) size |
| $S$ | Stride size |

#### nn.Conv2d 파라미터

```python
nn.Conv2d(
    in_channels=3,      # 입력 채널 수
    out_channels=64,    # 출력 채널 수
    kernel_size=3,      # 필터 크기 (3x3)
    stride=1,           # 이동 간격
    padding=1           # 패딩 크기
)
```

#### 예시

```python
sample = torch.randn(16, 3, 224, 224)  # [B, C, H, W]
conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
output = conv(sample)
print(output.shape)  # torch.Size([16, 6, 220, 220])

# 계산: (224 + 2×0 - 5) / 1 + 1 = 220
```

> ⚠️ `in_channels`는 입력 이미지의 채널 수와 **반드시 일치**해야 함!

---

### 1-2. Pooling Layer

Feature map의 크기를 줄이고 중요한 특징을 추출

| Pooling | 설명 | PyTorch |
|---------|------|---------|
| **Max Pooling** | 영역 내 최댓값 선택 | `nn.MaxPool2d(kernel_size=2)` |
| **Average Pooling** | 영역 내 평균값 선택 | `nn.AvgPool2d(kernel_size=2)` |

---

### 1-3. Custom CNN 모델 (MNIST 분류)

```python
class CNN(nn.Module):
    def __init__(self, num_classes, dropout_ratio):
        super(CNN, self).__init__()
        
        self.layer = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(1, 16, kernel_size=5),   # [B,1,28,28] → [B,16,24,24]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5),  # [B,16,24,24] → [B,32,20,20]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),       # [B,32,20,20] → [B,32,10,10]
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=5),  # [B,32,10,10] → [B,64,6,6]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),       # [B,64,6,6] → [B,64,3,3]
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(64 * 3 * 3, 100),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(100, num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layer(x)
        return x
```

#### CNN vs DNN 파라미터 비교

| 모델 | 파라미터 수 | 정확도 (MNIST) |
|------|------------|----------------|
| DNN | ~570,000 | 98.02% |
| CNN | ~70,000 | **99.23%** |

> CNN은 **파라미터가 적으면서도 성능이 더 좋음!**

---

## 2. VGG 모델

### 2-1. VGG 모델이란?

- 2014년 ImageNet 대회 **2위**
- **3×3 작은 필터**로 깊게 쌓는 전략
- 단순한 구조로 **변형이 쉬움**

#### VGG 구조 설정

```python
cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
# 숫자: out_channels, 'M': MaxPooling
```

### 2-2. VGG 모델 구현

```python
class VGG(nn.Module):
    def __init__(self, model_name, cfgs, in_channels=3, num_classes=10):
        super().__init__()
        self.conv_layers = self.create_conv_layers(cfgs[model_name])
        
        self.fcs = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU())
                in_channels = x
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x
```

### 2-3. VGG 성능 비교 (CIFAR10)

| 모델 | 파라미터 수 | Test Accuracy |
|------|------------|---------------|
| VGG11 | 28.1M | 78.61% |
| VGG13 | 28.3M | 80.58% |
| VGG16 | 33.6M | **81.34%** |
| VGG19 | 38.9M | 80.24% |

---

## 3. GoogLeNet 모델

### 3-1. GoogLeNet (Inception)이란?

- 2014년 ImageNet 대회 **1위**
- 핵심: **Inception Module**
- 다양한 필터 크기를 **병렬로 사용**

### 3-2. Inception Module

```
         Input
           │
    ┌──────┼──────┬──────┐
    │      │      │      │
  1×1    1×1    1×1   MaxPool
  conv   conv   conv    3×3
    │      │      │      │
    │    3×3    5×5    1×1
    │    conv   conv   conv
    │      │      │      │
    └──────┴──────┴──────┘
           │
     Concatenate
           │
        Output
```

#### 1×1 Convolution의 역할

1. **채널 수 감소** → 연산량 감소
2. **비선형성 추가** (ReLU와 함께 사용)

### 3-3. Inception Module 구현

```python
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, 
                 ch5x5_reduce, ch5x5, pool_proj):
        super().__init__()
        
        # Branch 1: 1×1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU()
        )
        
        # Branch 2: 1×1 → 3×3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(ch3x3_reduce),
            nn.ReLU(),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU()
        )
        
        # Branch 3: 1×1 → 5×5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(ch5x5_reduce),
            nn.ReLU(),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU()
        )
        
        # Branch 4: MaxPool → 1×1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU()
        )
    
    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)  # 채널 방향으로 concat
```

### 3-4. GoogLeNet 성능 (CIFAR10)

| 모델 | 파라미터 수 | Test Accuracy |
|------|------------|---------------|
| GoogLeNet | 5.9M | 84.13% |

> VGG보다 **파라미터가 적으면서 성능이 더 좋음!**

---

## 4. ResNet 모델

### 4-1. ResNet이란?

- **Degradation 문제 해결**: 레이어가 깊어질수록 오히려 성능이 떨어지는 문제
- 핵심: **Residual Learning (Skip Connection)**

```
        x
        │
    ┌───┴───┐
    │       │
  Conv      │ (identity)
  layers    │
    │       │
    └───┬───┘
        │
      x + F(x)  ← Residual Connection
        │
       ReLU
```

$$y = F(x) + x$$

### 4-2. Bottleneck Block

1×1 conv로 채널을 줄였다가 다시 늘리는 구조

```
Input (256 channels)
        │
    1×1 conv (64)   ← 채널 감소
        │
    3×3 conv (64)   ← 연산
        │
    1×1 conv (256)  ← 채널 복원
        │
   + ───┘ (skip connection)
        │
      Output
```

### 4-3. Bottleneck Block 구현

```python
class BottleNeck(nn.Module):
    expansion = 4  # 출력 채널 = out_channels × 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # 1×1 conv (채널 감소)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3×3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1×1 conv (채널 확장)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU()
        
        # Skip connection (차원 맞추기)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += self.shortcut(identity)  # Residual Connection
        out = self.relu(out)
        
        return out
```

### 4-4. ResNet 성능 (CIFAR10)

| 모델 | 파라미터 수 | Test Accuracy |
|------|------------|---------------|
| ResNet50 | 23.5M | 71.53% |

> ⚠️ CIFAR10 (32×32)은 ImageNet (224×224)보다 작아서 ResNet 성능이 낮게 나옴

---

## 모델 비교 정리

### 파라미터 수 & 성능 (CIFAR10)

| 모델 | 파라미터 | Accuracy | 특징 |
|------|----------|----------|------|
| Custom CNN | 70K | 99.23%* | 단순한 구조 |
| VGG16 | 33.6M | 81.34% | 3×3 conv 반복 |
| GoogLeNet | 5.9M | 84.13% | Inception Module |
| ResNet50 | 23.5M | 71.53% | Skip Connection |

*MNIST 기준

### 핵심 개념 요약

| 모델 | 핵심 아이디어 |
|------|--------------|
| **VGG** | 작은 필터(3×3)로 깊게 쌓기 |
| **GoogLeNet** | 다양한 필터 크기 병렬 사용 (Inception) |
| **ResNet** | Skip Connection으로 깊은 네트워크 학습 |

### 1×1 Convolution 활용

| 모델 | 용도 |
|------|------|
| GoogLeNet | 채널 감소 (연산량 절감) |
| ResNet | Bottleneck에서 채널 조절 |

---

## Reference
- [VGG 논문](https://arxiv.org/abs/1409.1556)
- [GoogLeNet 논문](https://arxiv.org/abs/1409.4842)
- [ResNet 논문](https://arxiv.org/abs/1512.03385)
- [PyTorch Conv2d 공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
