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
## 4. Container
1. 운영체제를 가상화한 형식으로, 각 컨테이너는 서로 격리된 공간에서 실행.
2. Virtual Machine과의 차이점: VM은 하드웨어까지 가상화. Container는 운영체제를 가상화하므로 휴대가 쉽고 가벼우며 효율적. 
3. 구성: Image(애플리케이션과 그 실행에 필요한 모든 파일을 포함한 불변의 템플릿), Registry(이미지가 저장되고 공유되는 곳), Container Runtime(컨테이너가 실행되기 위한 환경제공)
4. Docker: 컨테이너화 기술 사용하여 애플리케이션을 패키징하고 배포하는데 사용되는 오픈소스플랫폼
- Docker Image: 애플리케이션 실행하는데 필요한 모든 파일, 설정 포함하는 템플릿
- Docker Container: 이미지를 실행했을 때의 실행 Instance
- Docker Daemon: 이미지, 컨테이너 관리하는 백그라운드 서비스
- Docker Registry: 외부 이미지 저장소. 다른 사람들의 공유한 이미지 사용가능, private하게도 가능
- Docker Client: daemon과 상호작용하는 인터페이스

## 4-1. Container 실습
- labs.play-with-docker.com (설치없이 온라인 실습가능)
- https://hub.docker.com (docker hub account 생성)

1. 생성단계
- dockerfile 작성
- 파일 내 스크립트를 실행한다면 스크립트 작성
- docker 이미지 빌드
- docker 컨테이너 실행

```shell
FROM python:3.8-slim    # 기본 이미지로 파이썬3.8 사용
WORKDIR /app   # 작업 디렉토리 설정
COPY hello.py / app   # 파이썬 스크립트 복사
CMD [“python”, “./hello.py”]   # 스크립트 실행

docker build -t hello-world-python .   # 이미지 빌드
docker run hello-world-python   # 컨테이너 실행
```

2. 기본
```shell
cd ..
docker  # 도커 확인
clear 
mkdir practice_hellodockerworld  # 디렉토리 생성 
cd practice_hellodockerworld
ls
vim Dockerfile
FROM python:3.8-slim    # 기본 이미지로 파이썬3.8 사용
WORKDIR /app   # 작업 디렉토리 설정
COPY main.py / app   # 파이썬 스크립트 복사
CMD [“python”, “./main.py”]   # 스크립트 실행
ls
vim main.py
print(“Hello, Docker World!!!”)
print(sys.version_info)
cat main.py # 확인
ls

docker build -t hello-docker-world .   # 이미지 빌드 (.은 경로)
docker run hello-docker-world   # 컨테이너 실행

# 만약 여기서 파일 변경됐다면, 다시 빌드
docker build -t hello-docker-world .   # 이미지 빌드 (.은 경로)
docker run hello-docker-world   # 컨테이너 실행

docker ps # 실행중인 도커 리스트
docker stop [컨테이너 아이디]  # 실행중인 도커 중단
docker ps

docker rmi hello-docker-world -f  # 이미지 삭제
docker images  # 이미지 확인
```

3. Nginx기반
```shell
mkdir server_docker_images
docker search nginx  # 도커허브 웹사이트에서도 public images검색가능
docker pull nginx # 다운로드
docker images # 다운받은 이미지 확인
docker run -it -d -p 8001:80 --name nginxserver nginx:latest &
docker ps  # 실행 확인
docker stop [컨테이너아이디]
docker start nginxserver # 이미 run한 이미지를 stop했다가 다시 실행할때

# 1) docker내 파일을 직접 수정
docker exec -it [컨테이너 아이디] bash
ls
cd /usr/share/nginx/html  # html 파일 수정하기
ls
cat index.html
Echo “fastcampus’ nginx server” > index.html
cat index.html
exit

# 2) 파일 생성 후 컨테이너로 옮기기
ls
cd server_docker_container
ls
vim index.html # 작성
cat index.html 
docker ps
docker cp index.html nginxserver:/usr/share/nginx/html/index.html # 파일을 해당경로로 복사
```

4. Scikit-learn 기반
```shell
mkdir scikitlearndocker
cd scikitlearndocker
ls
vim Dockerfile
# 파일 작성
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY model_learn.py /app/   # py파일을 app 디렉토리로 복사
CMD [“python”, “./model_learn.py”]

cat Dockerfile # 파일 저장 후 내용 확인

vim requirements.txt
scikit-learn
cat requirements.txt
vim model_learn.py
# 필요 라이브러리, 모델링 코드 작성(전처리, 분할, 학습, 평가 등)
cat model_learn.py

docker build -t scikitlearn_modellearn .    # 빌드
docker run --name mldevelopment scikitlearn_modellearn
```
```shell
docker images
# 로컬에서 만든 이미지를 도커허브 repository로 저장
docker tag scikitlearn_modellearn:latest [나의 도커허브 아이디]/scikitlearn_modellearn:latest
docker images
docker push [나의 도커허브 아이디]/scikitlearn_modellearn:latest
clear
```
```shell
# 나의 repositoty에 있는 이미지 로컬로 다운받기
docker pull [도커허브에서 복사한 해당이미지 이름/태그]
docker images  # 다운받은 이미지 확인
docker run --name test [해당이미지 이름/태그]
```
















