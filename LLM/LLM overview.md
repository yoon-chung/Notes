# [LM to LLM] Large Language Model 이란?
---

## 1. Large Language Model 개요

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

## 2. Large Language Model의 방향성

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

---
## 3. LLM 이론

### 3.1 In-Context Learning (ICL)

#### Fine-Tuning vs In-Context Learning

| 구분 | Fine-Tuning | In-Context Learning |
|---|---|---|
| 파라미터 업데이트 | O (전체 또는 일부) | X |
| 학습 데이터 필요량 | 상대적으로 많음 | 0~수 개의 예시 |
| 메모리 방식 | Parametric memory | Non-parametric memory |
| 적용 방식 | 모델 가중치 변경 | 프롬프트에 예시 포함 |

#### N-Shot Learning

- **Zero-shot**: 예시 없이 지시문만으로 태스크 수행. GPT-2 논문에서 독해, 번역, 요약 등에서 상당한 성능을 보임.
- **One-shot**: 하나의 예시 + 지시문 제공. 사람의 소통 방식과 가장 유사.
- **Few-shot**: 여러 예시 제공. 태스크 특화 데이터 필요성을 크게 줄여줌.

**핵심 인사이트**: 모델 크기가 클수록, 예시 수가 많을수록 ICL 성능이 향상됨. Chain-of-Thought prompting처럼 프롬프트 설계도 성능에 큰 영향을 미침.

---

### 3.2 ChatGPT 학습 방법

#### 3단계 학습 파이프라인

##### Step 1: SFT (Supervised Fine-Tuning)
- 지시문(prompt)과 이상적인 응답 쌍으로 구성된 demonstration dataset(약 13K) 구축
- 라벨러가 프롬프트에 적합한 응답을 직접 작성
- 이 데이터로 GPT-3를 fine-tuning → SFT 모델 생성
- 지시를 따르는 능력이 기존 GPT-3보다 향상되나 완벽하지는 않음

##### Step 2: Reward Model (RM) 학습
- SFT 모델이 하나의 프롬프트에 대해 여러 응답(4~9개)을 생성
- 라벨러가 응답들에 대해 선호도 순위를 매김 (comparison dataset, 약 33K)
- 이 데이터로 사람의 선호도를 예측하는 보상 모델을 학습

##### Step 3: RLHF (Reinforcement Learning from Human Feedback)
- PPO(Proximal Policy Optimization) 알고리즘 사용
- SFT 모델이 응답을 생성 → RM이 보상 점수를 부여 → 보상을 최대화하는 방향으로 정책 업데이트
- 결과적으로 사람이 선호하는, 유용하고 안전한 응답을 생성하도록 최적화

#### Instruction Tuning
- 기존 LM은 다음 토큰 예측에 최적화되어 있어 사람의 지시를 잘 따르지 못함
- Instruction Tuning은 명령을 따르도록 모델을 fine-tuning하는 방식
- InstructGPT(2022.01)에서 제안된 실험 방식이 ChatGPT에 반영됨

#### 프롬프트 활용 팁
- **Persona Injection**: 역할 부여 (예: "너는 면접관이다")
- **프롬프트 4요소**: 지시사항, 참고 데이터, 출력 형식, 사용자 입력
- **3Cs Framework**: Clarity(명확성), Context(맥락), Constraints(제약조건)
- 짧고 간결하게 작성하고, 출력 형태를 지정하며, 구역을 나누어 복잡한 프롬프트를 구조화

---

### 3.3 Parameter Efficient Fine-Tuning (PEFT)

#### PEFT가 필요한 이유
- LLM이 커지면서 전체 파라미터를 fine-tuning하는 것이 하드웨어적으로 불가능해짐
- Fine-tuned 모델의 크기가 원본과 동일하여 저장/배포 비용이 큼
- Catastrophic forgetting 문제 완화 가능

#### 주요 PEFT 기법들

##### Prefix-Tuning
- 각 Transformer 레이어 입력 앞에 학습 가능한 task-specific 벡터(prefix)를 추가
- LM 파라미터는 고정, prefix만 학습
- 하나의 LM으로 prefix를 바꿔가며 여러 태스크 처리 가능

##### Prompt Tuning
- 입력 텍스트 앞에 추가되는 k개의 soft token embedding만 학습
- 모델 전체는 freeze 상태 유지
- 모델 크기가 클수록 full fine-tuning과 성능 차이가 줄어듦

##### P-Tuning
- Prompt Encoder(Bi-LSTM)를 사용해 continuous prompt embedding 생성
- Anchor token을 추가하여 성능 개선
- GPT 스타일 모델에서도 BERT 수준의 NLU 성능 달성

##### LoRA (Low-Rank Adaptation)
- 사전학습 가중치를 고정하고, 저랭크 행렬 분해를 통한 어댑터만 학습
- `h = W₀x + BAx` (B, A가 저랭크 행렬)
- 추론 시 추가 지연 없음 (기존 가중치에 합산 가능)
- GPT-3 175B 기준 전체의 0.01% 파라미터만으로 유사 성능 달성

##### QLoRA (Quantized LoRA)
- 사전학습 모델을 4-bit로 양자화하여 저장
- LoRA 어댑터는 16-bit로 학습 유지
- 단일 48GB GPU에서 65B 모델 fine-tuning 가능
- 16-bit full fine-tuning과 거의 동일한 성능

##### IA3
- Attention의 Key, Value와 FFN 출력에 학습 가능한 rescaling 벡터를 적용
- LoRA보다 더 적은 파라미터로 높은 성능
- GPT-3 ICL보다도 우수한 결과

##### LLaMA-Adapter
- 상위 Transformer 레이어에만 학습 가능한 프롬프트 토큰 삽입
- 1.2M 파라미터, 1시간 학습으로 instruction-following 능력 부여
- Visual encoder와 결합하여 멀티모달 지원 가능

#### Quantization (양자화)
- 모델 파라미터를 낮은 비트(예: FP32 → INT8)로 변환하는 경량화 기법
- 주 목적: 추론 시간 및 메모리 사용량 감소
- 신경망 파라미터가 정규분포를 따르는 특성을 활용하여 분위수 기반 양자화 수행

---

