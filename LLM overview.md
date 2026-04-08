# [LM to LLM] Large Language Model 이란?
---

## 01. Large Language Model 개요

### 1.1 LLM의 정의

- **LLM이란**: 기존 언어모델의 확장판으로, 방대한 파라미터 수를 가진 언어모델
- LLM 시대 = **Foundation Model**의 시대: Machine Learning → Deep Learning → Foundation Models로 진화
- **창발성(Emergence)**: 단일 모델로 텍스트 생성, 요약, 정보 추출, QA, 챗봇 등 여러 Task를 처리 가능
- 과거에는 각 Task마다 개별 모델을 학습했지만, 이제는 하나의 Foundation Model을 pretrain 후 adapt하는 방식
- **Human Alignment**이 핵심
  - Stage 1: 대규모 비정형 데이터로 사전학습 (비용 높음, 정렬 안 됨)
  - Stage 2: Instruction Tuning으로 지시를 따르도록 정렬 (비용 낮음, 정렬 발생)
  - 정렬은 기본 지식이 충분히 학습된 이후에야 효과적

### 1.2 LLM의 등장 배경

- **Scaling Law**: 모델 크기, 데이터셋 크기, 연산량을 함께 늘리면 성능이 멱법칙(power-law)에 따라 향상
- **In-Context Learning**: Zero-shot, One-shot, Few-shot으로 gradient 업데이트 없이 프롬프트만으로 새로운 태스크 수행
  - 모델이 클수록 In-Context Learning 능력이 급격히 상승 (특히 175B 파라미터)
- **Emergent Abilities**: 특정 규모를 넘어서면 수학, 추론 등의 능력이 급격히 향상
- **Instruction Tuning**: 모델 크기에 관계없이 held-out 태스크에서 zero-shot 성능을 크게 향상시킴

### 1.3 LLM의 제작 프로세스

**필요 재료:**
- **Infra**: 하이퍼스케일 클라우드, 슈퍼컴퓨팅, 대규모 데이터센터
- **Backbone Model**: 기반이 되는 사전학습 모델
- **Tuning**: 경량화 및 효율적 튜닝 기술 (양자화 등)
- **Data**: 고품질 대량 학습 데이터 (Prompt, Instruction)

**데이터 구성**: 모델마다 Webpages, Books & News, Code, Scientific Data, Conversation Data 등의 비율이 다름

**데이터 전처리 파이프라인:**
1. Quality Filtering (언어, 메트릭, 통계, 키워드 기반)
2. De-duplication (문장/문서/셋 레벨)
3. Privacy Reduction (PII 탐지 및 제거)
4. Tokenization (SentencePiece, Byte-level BPE 등)

**결과물:**
- **Base LLM**: 다음 단어를 예측하는 모델 (텍스트 완성)
- **Instruction Tuned LLM**: 지시를 따르는 모델 = Base LLM + SFT + RLHF

---

## 02. Large Language Model의 방향성

### 2.1 Data & Size

- **Chinchilla 논문**: 모델 크기만 키우는 것이 아니라, 데이터도 비례하여 늘려야 compute-optimal
  - Chinchilla(70B, 1.4T 토큰)가 Gopher(280B, 300B 토큰)를 능가
- 모델링보다 **데이터 구성**이 더 중요할 수 있음
- 사전학습 데이터의 도메인 구성에 따라 In-Context Learning 성능이 크게 달라짐
- **사전학습 모델의 크기가 중요**: Small LLM의 모방 학습은 스타일만 흉내내고 실질적 능력 향상은 제한적
- **LIMA 논문**: 사전학습에서 거의 모든 지식이 학습되며, 소량의 고품질 instruction 데이터만으로도 높은 성능 달성 가능

### 2.2 Multimodal

- 언어뿐 아니라 비전, 오디오 등 다양한 모달리티를 통합하는 방향
- **주요 모델들**:
  - PaLM-E: 로봇 제어 + 멀티모달 언어 모델
  - Kosmos-1 & 2: Microsoft의 멀티모달 LLM (언어 + 비전 + 오디오)
  - GPT-4V: 이미지 입력을 받아 텍스트 출력
  - Gemini: Google DeepMind의 텍스트/이미지/비디오/오디오/코드 통합 모델
  - ImageBind: Meta의 6개 모달리티 통합 임베딩 공간

### 2.3 Multilingual

- **오픈소스**: BLOOM (176B, 59개 언어), NLLB (200+ 저자원 언어 번역)
- **Google PaLM 2**: 다국어 번역에서 Google Translate를 능가하는 성능
- **GPT-4**: 95개 언어 지원, MMLU 벤치마크에서 대부분 언어 80%+ 정확도

### 2.4 Synthetic Data

- 2030년까지 합성 데이터가 실제 데이터를 넘어설 전망
- LLM을 이용한 레이블링 성능 변천사:
  - 2021년: GPT-3는 사람보다 낮음
  - 2023년 3월: GPT-3.5는 인간과 거의 동등
  - 2023년 4월: GPT-4는 시간당 $25 크라우드워커보다 우수

### 2.5 Domain Specialized

- 일반 데이터로 학습한 모델을 도메인 데이터로 재학습하여 특정 도메인 모델 생성
- 예: 네이버 HyperCLOVA X (한국어 특화, 검색·코딩·번역·요약·상담·추천 등)

### 2.6 Evaluation

- **HELM**: Accuracy, Robustness, Fairness, Bias, Toxicity 등 다차원 평가
- **벤치마크**: SuperGLUE, Big Bench Hard, MMLU, TriviaQA, GSM8K, HumanEval 등
- **LLM-as-Judge**: GPT-EVAL, LLM-Eval 등 LLM을 평가자로 활용하는 방식 등장

### 2.7 Prompt Engineering

- **Prompt**: LLM에게 원하는 결과를 얻기 위한 입력/지시
- **구성요소**: Instruction, Context, Input Data, Output Indicator

**주요 기법:**
- **Chain-of-Thought (CoT)**: 답변 도출 과정(추론 단계)을 함께 생성하여 성능 향상
- **Zero-shot CoT**: "Let's think step by step"만으로도 추론 성능 향상
- **Prompt Manager**: 개별 모달리티/API를 연결하는 관리 기술 (Visual ChatGPT, Toolformer 등)
- **Function Calling**: 모델이 API 호출 시점과 파라미터를 JSON으로 출력
- **PEFT**: P-Tuning, LoRA 등 일부 파라미터만 튜닝하여 효율적 성능 유지
- **Parameter Tuning**: Temperature, Top_p, frequency_penalty, presence_penalty 조절
- **Auto-GPT / Voyager**: 목표만 설정하면 자율적으로 반복 실행하여 결과 도출

### 2.8 3rd Party Platform

- LLM 기반 앱/서비스가 폭발적으로 증가 (Slack, 카카오톡 챗봇, PDF 챗봇 등)
- **AI Tech Stack**: Hardware → Cloud → Foundation Model → Apps
- **Private AI**: RAG(검색 증강 생성)를 통해 기업 내부 지식 기반 LLM 활용
- **DevOps → MLOps → LLMOps(FMOps)**: API 기반 접근, 프롬프트 엔지니어링 중심 성능 향상

### 2.9 Open Source

- GPT-3 규모(175B)까지 모델/코드 공개됨 (Meta OPT, BLOOM 등)
- EleutherAI: Big Model 민주화 추구
- **LLaMA 이후 오픈소스 생태계 폭발**:
  - llama.cpp로 라즈베리파이에서도 7B 모델 구동 가능
  - Alpaca, Vicuna, KoAlpaca, KULLM 등 파생 모델 등장
- **Open LLM Leaderboard**: HuggingFace에서 오픈 모델 성능 추적/비교

### 2.10 To be

- **빠른 적응(Rapid Adaptation)**: 검색 패러다임 변화 (키워드 → Instruct 방식)
- **LLM의 약점 공략**: Reasoning, Commonsense, Hallucination, Expert Knowledge, Ethics
