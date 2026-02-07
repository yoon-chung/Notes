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
