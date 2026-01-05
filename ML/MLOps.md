# MLOps
ML Life cycle을 체계적으로 관리, 적절한 인프라에 의존(데이터 처리, 모델 트레이닝, 배포, 모니터링과 같은 핵심 작업 지원) 
## 1. 인프라 요소
1. Storage: 데이터 저장, 백업, 복구 기능 제공
- Amazon S3 대규모 데이터를 저장하고 접근하는 클라우드 기반 서비스, 대규모 이미지 데이터셋을 s3 버킷에 저장하여 ML모델 트레이닝에 활용.
2. Computing Resources: 
- Google Cloud의 Computing Engine에서 GPU를 활용해 모델 빠르게 트레이닝
3. 환경관리 툴
- 프로젝트별 독립적 환경 제공하는 패키지/환경관리시스템
- Conda 환경을 사용해 팀내 다양한 ML프로젝트 의존성 관리
4. Container
- 애플리케이션과 그 의존성을 패키지화하여 일관된 환경에서 실행할 수 있도록 지원
- Docker 컨테이너를 사용해 모델 트레이닝 환경을 일관되게 유지
5. Orchestrator
- 여러 컨테이너의 배포, 확장, 네트워킹 관리
- Kubernetes를 사용해 여러 모델 서빙 컨테이너를 자동으로 스케일링 및 관리
6. Workflow Management
- Apache Airflow를 사용해 일일 데이터 처리 및 모델 트레이닝 작업 자동화
7. CI/CD
- 모델 개발 및 테스트 주기 단축시켜 빠른 반복을 가능하게 하는 도구
8. 버전 관리
- 이전에 개발된 모델 재현을 위해 원하는 버전의 데이터, 코드를 활용
9. HTTP, REST API
- 다른 시스템과의 통신을 위한 표준 프로토콜 및 인터페이스
- 예: REST API call 요청을 해서 해당 모델 트레이닝 컨테이너를 실행

## 2. Storage & Computing Resources
1. Storage 
- Amazon S3, Cloud Storage, Azure Blob Storage
- 대규모 이미지/비디오 데이터셋의 저장, 분석, 글로벌 ML 파이프라인에서의 데이터 공유
- 분산파일시스템: Hadoop Distributed File System, GlusterFS
- 데이터 웨어하우스: Snowflake, Amazon Redshift, Google BigQuery
- 데이터 레이크: AWS Lake Formation, Azure Data Lake
2. Computing
- 클라우드 기반: AWS EC2, Google Compute Engine
- GPU/TPU: NVIDIA의 Tesla 시리즈 GPU, Google Cloud TPU
- 서버리스 컴퓨팅: AWS Lambda, Google Cloud Functions, Azure Functions
- 컨테이너화된 컴퓨팅: Docker 컨테이너, Kubernetes 클러스터

## 3. 환경관리툴
프로젝트별로 격리된 개발환경 설정 유지
1. Conda: 파이썬 등 프로그래밍 언어를 위한 환경 관리 시스템 (conda forge, Anaconda Repository 등 채널 통해 수천개 패키지 접근 가능)
2. Virtualenv: 파이썬의 가상환경 생성도구 (pip사용)
3. Pipenv: pip + virtualenv 기능 결합 (pipfile 사용)

## 3-1. Conda 실습
```shell
conda
conda --version
clear
```
```shell
conda create --name fastcampus_mlops_env
y
conda activate fastcampus_mlops_env
conda deactivate
```
```shell
conda create --name fastcampus_mlops_python_env python=3.8  # 선택환경에 특정버전 설치
y
python –version  # 버전 확인
exit()
```
```shell
conda env list # 동료의 env 환경이름 확인
conda activate fastcampus_mlops_env # 해당 env 실행
conda env export > environment_my.yml # env 환경내 설치내용 저장
cat environment_my.yml # 설치내용 확인
conda deactivate
```
```shell
# 한번에 env 복사
conda env create --name newone -f environment_my.yml  # newone이라는 env 만들기
conda activate newone # newone env 실행
conda env export > environment_test.yml # env 환경내 설치내용 저장 
cat environment_test.yml # 설치내용 확인
conda deactivate
```

## 3-2. Virtualenv 실습
```shell
pip install virtualenv # 설치
virtualenv # 확인
virtualenv myenv
source myenv/bin/activate
pip install numpy pandas
python
exit()
```
```shell
virtualenv myenv2 -p /user/bin/python3.8 # 원하는 파이썬 버전으로
pip install requests flask # 예시
source myenv/bin/activate 
ls # 확인
clear
```













