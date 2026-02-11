# PyTorch 모니터링 - TensorBoard & WandB

## 목차
1. [TensorBoard](#1-tensorboard)
2. [WandB](#2-wandb)
3. [WandB Sweep (하이퍼파라미터 튜닝)](#3-wandb-sweep)

---

## 모니터링이 필요한 이유

```
┌─────────────────────────────────────────────────────────────────┐
│                    딥러닝 모니터링 목적                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. 학습 진행 확인                                              │
│      └─ Loss/Accuracy가 정상적으로 변화하는지                    │
│                                                                 │
│   2. 문제 조기 발견                                              │
│      └─ Overfitting, Gradient Vanishing/Exploding 감지          │
│                                                                 │
│   3. 실험 비교                                                   │
│      └─ 여러 하이퍼파라미터 설정의 성능 비교                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. TensorBoard

### 1-1. 설치 및 기본 사용

```bash
pip install tensorboard
```

```python
from torch.utils.tensorboard import SummaryWriter

# 로그 디렉토리 지정
writer = SummaryWriter("./runs/experiment1")
```

### 1-2. 주요 기능

| 기능 | 메서드 | 용도 |
|------|--------|------|
| **Scalars** | `add_scalar()` | Loss, Accuracy 등 추적 |
| **Graphs** | `add_graph()` | 모델 구조 시각화 |
| **Histograms** | `add_histogram()` | Weight/Bias 분포 |
| **Images** | `add_image()` | 이미지 시각화 |
| **Embeddings** | `add_embedding()` | 고차원 벡터 시각화 |

### 1-3. 코드 예시

#### Scalar 로깅 (Loss, Accuracy)

```python
# 학습 루프 내에서
writer.add_scalar("Loss/train", train_loss, epoch)
writer.add_scalar("Loss/valid", valid_loss, epoch)
writer.add_scalar("Accuracy/train", train_acc, epoch)
writer.add_scalar("Accuracy/valid", valid_acc, epoch)
```

#### 모델 그래프

```python
writer.add_graph(model, input_tensor)
```

#### Histogram (Weight 분포)

```python
for name, param in model.named_parameters():
    writer.add_histogram(name, param, epoch)
```

#### 이미지 로깅

```python
import torchvision

img_grid = torchvision.utils.make_grid(images)
writer.add_image('Sample Images', img_grid)
```

#### 임베딩 시각화

```python
writer.add_embedding(
    mat=embeddings,      # [N, D] 임베딩 행렬
    metadata=labels,     # 라벨 리스트
    label_img=images     # 이미지 (선택)
)
```

### 1-4. TensorBoard 실행

```bash
# 터미널
tensorboard --logdir ./runs

# Jupyter/Colab
%load_ext tensorboard
%tensorboard --logdir ./runs
```

### 1-5. 학습 코드에 통합

```python
def training_loop(model, train_loader, valid_loader, ...):
    writer = SummaryWriter("./runs/exp1")
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(...)
        valid_loss, valid_acc = evaluate(...)
        
        # 로깅
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/valid", valid_loss, epoch)
        writer.add_scalar("Accuracy/valid", valid_acc, epoch)
    
    writer.close()  # 🔑 반드시 종료
```

---

## 2. WandB

### 2-1. 설치 및 로그인

```bash
pip install wandb
```

```python
import wandb

wandb.login()  # API key 입력 (최초 1회)
```

### 2-2. TensorBoard vs WandB

| 구분 | TensorBoard | WandB |
|------|-------------|-------|
| 저장 위치 | 로컬 파일 | 클라우드 (웹) |
| 협업 | 어려움 | 쉬움 (링크 공유) |
| 여러 서버 | 각각 확인 | 한 곳에서 통합 |
| 하이퍼파라미터 튜닝 | 미지원 | **Sweep 지원** |

### 2-3. 기본 사용법

```python
# 실험 시작
run = wandb.init(
    project='my-project',    # 프로젝트명
    name='experiment-1',     # 실험명
    config={                 # 하이퍼파라미터
        'lr': 0.001,
        'batch_size': 32,
        'epochs': 100
    }
)

# 로깅
run.log({'loss': 0.5, 'accuracy': 0.85}, step=epoch)

# 실험 종료
run.finish()  # 🔑 반드시 종료
```

### 2-4. 주요 기능

#### Config 접근

```python
# wandb.config로 하이퍼파라미터 접근
lr = wandb.config.lr
batch_size = wandb.config.batch_size
```

#### 이미지 로깅

```python
run.log({
    'images': [wandb.Image(img, caption=str(label)) 
               for img, label in zip(images, labels)]
})
```

#### 모델 자동 추적 (watch)

```python
# Weight, Bias, Gradient 자동 로깅
run.watch(model, criterion, log='all', log_graph=True)
```

| log 옵션 | 설명 |
|----------|------|
| `'gradients'` | Gradient만 |
| `'parameters'` | Weight/Bias만 |
| `'all'` | 모두 로깅 |

### 2-5. 학습 코드에 통합

```python
def training_loop(model, train_loader, valid_loader, ...):
    run = wandb.init(project='mnist', name='exp1')
    run.watch(model, criterion, log='all')
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train(...)
        valid_loss, valid_acc = evaluate(...)
        
        # 로깅
        run.log({
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'valid_accuracy': valid_acc
        }, step=epoch)
    
    run.finish()
```

---

## 3. WandB Sweep

### 3-1. Sweep이란?

하이퍼파라미터 조합을 **자동으로 탐색**하는 기능

```
┌─────────────────────────────────────────────────────────────────┐
│                      Sweep 프로세스                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. Config 정의                                                │
│      └─ 탐색할 하이퍼파라미터 범위 지정                          │
│                                                                 │
│   2. Sweep 생성                                                 │
│      └─ wandb.sweep() → sweep_id 생성                          │
│                                                                 │
│   3. Agent 실행                                                 │
│      └─ wandb.agent()가 자동으로 실험 수행                      │
│                                                                 │
│   4. 결과 확인                                                   │
│      └─ 웹에서 최적 하이퍼파라미터 확인                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3-2. 탐색 방법 (method)

| Method | 설명 | 특징 |
|--------|------|------|
| `grid` | 모든 조합 탐색 | 완전 탐색, 시간 오래 걸림 |
| `random` | 랜덤 선택 | 빠름, 운에 의존 |
| `bayes` | 베이지안 최적화 | 효율적, 이전 결과 활용 |

### 3-3. Sweep Config 작성

```python
sweep_config = {
    'method': 'random',  # 탐색 방법
    'metric': {
        'goal': 'maximize',    # 또는 'minimize'
        'name': 'valid_accuracy'
    },
    'parameters': {
        'lr': {
            'min': 0.0001,
            'max': 0.01
        },
        'dropout': {
            'values': [0.1, 0.2, 0.3]  # 이산값
        },
        'batch_size': {
            'values': [16, 32, 64]
        }
    }
}
```

### 3-4. Sweep 실행

```python
# 1. 학습 함수 정의
def run_sweep():
    run = wandb.init()
    
    # config에서 하이퍼파라미터 가져오기
    model = MyModel(dropout=wandb.config.dropout)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.lr)
    
    # 학습 수행
    for epoch in range(num_epochs):
        train_loss = train(...)
        valid_acc = evaluate(...)
        run.log({'valid_accuracy': valid_acc})

# 2. Sweep 생성
sweep_id = wandb.sweep(sweep_config, project='my-project')

# 3. Agent 실행 (count: 실험 횟수)
wandb.agent(sweep_id, function=run_sweep, count=10)
```

---

## 핵심 코드 패턴

### TensorBoard 템플릿

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./runs/exp1")

for epoch in range(num_epochs):
    # 학습/평가
    train_loss, train_acc = train(...)
    valid_loss, valid_acc = evaluate(...)
    
    # Scalar 로깅
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/valid", valid_loss, epoch)
    
    # Histogram 로깅 (선택)
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

writer.close()
```

### WandB 템플릿

```python
import wandb

run = wandb.init(project='project-name', name='exp1', config={...})
run.watch(model, log='all')

for epoch in range(num_epochs):
    train_loss, train_acc = train(...)
    valid_loss, valid_acc = evaluate(...)
    
    run.log({
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'valid_accuracy': valid_acc
    }, step=epoch)

run.finish()
```

---

## Reference
- [TensorBoard - PyTorch 공식 문서](https://pytorch.org/docs/stable/tensorboard.html)
- [WandB 공식 문서](https://docs.wandb.ai/)
- [WandB Sweep 가이드](https://docs.wandb.ai/guides/sweeps)
