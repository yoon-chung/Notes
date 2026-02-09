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
