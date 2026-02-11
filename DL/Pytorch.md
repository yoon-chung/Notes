# Pytorch
## 1. 환경설정
### 1. Anaconda 설치
- Anaconda 홈피에서 리눅스 설치를 클릭 linux에서 ‘64-Bit (x86) Installer’를 우클릭해 링크 주소 복사
- 터미널: bash파일 설치- “wget [링크주소]” => 설치파일 실행-”bash [파일명.sh]“ => 터미널 재실행하면 (base)표시됨
### 2. Pytorch 설치
#### 1. NVIDIA GPU 없는 PC 
- 아나콘다에서 새로운 가상환경 만들기: conda create --name pytorch_test --clone base 
- conda activate pytorch_test
- conda install pytorch==2.0.0 torchvision==0.15.0 cpuonly -c pytorch
- 설치 완료후 확인: import tourch
#### 2. NVIDIA GPU 있는 PC 
- GPU정보 확인: nvidia-smi
- GPU이름 확인: nvidia-smi -L
- CUDA 위키피디아 https://en.wikipedia.org/wiki/CUDA 에서 GPU이름에 맞는 compute capability(version) 찾기
- 파이토치 버전별 설치페이지 https://pytorch.org/get-started/previous-versions/ 에서 버전선택, 설치
- 설치 완료후 확인: torch.cuda.is_available()

## 2. Tensor (텐서)
- 데이터 배열(array)을 의미
### 1. Broadcasting
- 차원이 다른 두 텐서 혹은 텐서-스칼라 간 연산 가능하게 해줌
### 2. Sparse Tensor
- Dense tensor: 배열의 모든 위치에 값 가짐. 많은 계산량/시간 증가
- Sparse tensor: 0이 아닌 원소와 그 위치를 저장하는 텐서
- COO방식: (row_index, column_index, value) 의 형태로 저장. 비효율적 메모리 사용
- CSR/CSC방식: (row_pointer, column_index, value) / (column_pointer, row_index, value) 형태. 중복 저장 줄어들어 효율적 메모리 사용

## 3. 딥러닝을 위한 Pytorch
- 딥러닝 학습단계: Data > Model > Output > Loss > Optimization
- torch.utils.data.Dataset, torch.utils.data.DataLoader
- torch.nn.Module
- torch.nn, torch.nn.functional
- torch.optim
### 1. Data
- Dataset, DataLoader 사용하여 데이터 로드: 데이터셋에서 미니배치 크기의 데이터를 반환
- Dataset: 단일 데이터를 처리하여 반환하는 작업 수행(tensor로). 

```
# Custom dataset 구현 (모든 반환 데이터의 차원 크기는 같아야함)
from torch.utils.data import Dataset

Class CustomDataset(Dataset):
	def __init__(self):
		pass
	def __getitem__(self, idx):
		pass                         # 반드시 tensor형태로 반환되어야함 
	def __len__(self):
		pass
```

- DataLoader: 데이터를 미니배치로 묶어서 반환
- batch_size, shuffle(epoch마다 데이터 순서가 섞이는지), num_workers(서브 프로세스 개수), drop_last(마지막 미니 배치 데이터 수가 미니배치 크기보다 작은 경우, 데이터 버릴지 결정)

### 2. Model
- Torchvision: 이미지 특화. torchvision.models.\[model 이름\]()
- PyTorch Hub: torch.hub.load()
- Custom Model: 
```
class CustomModel(nn.Module):
	def __init__(self):                 # 부모클래스, 모델 레이어, 파라미터 초기화
		super().__init__() 
		self.encoder=nn.Linear(10, 2)
		self.decoder=nn.Linear(2, 10)
	def forward(self, x):              # 입력 데이터에 대한 연산 정의
		out = self.encoder(x)
		out = self.decoder(out)
		Return out
model = CustomModel()
```

### 3. 역전파, 최적화
#### 1. 기본 구조
- optimizer.zero_grad(): 이전 gradient를 0으로 설정 (else 계속 누적됨)
- output = model(data): 데이터를 모델 통해 연산
- loss = loss_function(output, label): loss값 계산
- loss.backward(): AutoGrad통해 자동계산됨
- optimizer.step(): 계산된 gradient 사용해 각 파라미터 업뎃
#### 2. AutoGrad
- 연산에 대한 미분을 자동계산하기 위해 computational graph를 생성(노드, 엣지로 표현)
- computational graph와 chain rule에 따라 gradient를 자동으로 계산해줌

### 4. 추론, 평가
- model.eval(): 모델을 eval모드로 전환. 특정 레이어들이 학습/추론에서 각각 다르게 작동해야하기때문
- torch.no_grad(): AutoGrad기능 비활성화 
```
model.eval()
with torch.no_grad():
	for data in test_dataloader:
		pred = model(data)
```
- 예측결과와 실제 라벨 비교하여 모델 성능 평가

## 4. 전이학습
### 1. 개념
- Pretrained Model: 대규모 데이터셋 기반으로 학습된 모델, 학습 task에 대한 일반적 지식 가짐 (최근 GPT, PALM등)
- 전이학습: pretrained model 지식을 다른 task에 활용. 일반적 지식 기반으로 더 효과적으로 새로운 지식 학습 가능
- Fine-Tuning: 전이학습의 한 방법. Pretrained model를 그대로 or layers를 추가한 후 새로운 작업에 맞는 데이터로 모델을 추가로 더 훈련시키는 방법
- Domain adaptation: 전이학습의 한 방법. 도메인은 데이터가 속하는 분포. A도메인에서 학습한 모델을 B도메인으로 전이하여 도메인 간 차이를 극복하는 목적. (예) A:실제사진, B:애니메이션
### 2. 전이학습 전략
- 도메인이 비슷하고 dataset 크기 작을때 - 마지막 classifier만 추가 학습 (나머지 freeze)
- 도메인 비슷하고 dataset 크기 클때 - classifier + 다른 일부 layers로 추가 학습
- 도메인이 매우 다르고 dataset 크기 작으면 - 전이학습 부적합
- 도메인이 매우 다르고 dataset 크기 꽤 클때 - pretrained model의 꽤 많은 layers를 학습해야함
- learning rate: 작게 학습 (pretrained model의 일반적 지식을 크게 업뎃하지 않기위해)

### 3. 모델 커뮤니티 
- Timm for CV: CV분야에서 사용하는 사전 학습 모델 라이브러리
- Hugging Face for NLP, CV: 초기 NLP위주 -> 다양한 분야 사전학습 모델 라이브러리 제공


