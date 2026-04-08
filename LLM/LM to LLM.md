# LM to LLM 

---

## Ch1. 언어모델이란 무엇인가?

### 자연언어와 언어모델
- 언어모델: 인간의 자연어(문자)를 신경망이 처리할 수 있는 숫자로 변환하는 체계
- 모델 크기가 급격히 증가하는 추세 (ELMo 94M → GPT-3 175B → PaLM 540B)

### 언어모델 발전 타임라인
| 시대 | 주요 발전 |
|------|---------|
| 1950s | 번역에 대한 관심, 촘스키의 통사구조, 생성문법 |
| 1990s | 통계적 모델, RNN/LSTM |
| 2000s | 언어 모델링, Word Embedding, Google Translate |
| 2010s | Word2Vec, Encoder-Decoder, Attention, Transformer, Pre-trained 모델 |
| 2020s | GPT-3, LLM 시대 |

### Encoder vs Decoder
- **Encoder (BERT 계열)**: 양방향 문맥 이해, 분류/NER 등에 강점
- **Decoder (GPT 계열)**: 단방향 생성, 텍스트 생성에 강점
- **Encoder-Decoder (T5 계열)**: 번역, 요약 등 seq2seq 태스크

### LLM의 등장
- 전통 ML: 태스크별 labeled data → 개별 모델 학습 → 배포
- Foundation Model: 대규모 unlabeled data → 사전학습 → 다양한 태스크에 적응(Adapt)
- GPT-4는 다양한 시험에서 인간 수준 이상의 성능 달성

---

## Ch2. 전통적인 언어지식 표현 체계

### 정보이론 기초
- **Entropy**: 정보의 불확실성 측정. 높을수록 무질서
- **Cross-Entropy Loss**: 모델 예측 분포와 실제 분포 간의 차이 측정
- **KL Divergence**: 두 확률분포 간의 비대칭적 거리

### One-hot Encoding
- 어휘 크기 N의 벡터에서 해당 단어 위치만 1, 나머지 0
- 한계: 단어 간 유사도 표현 불가, 고차원 희소 벡터

### 통계적 언어모델
- 단어 시퀀스의 확률: P(W) = ∏ P(wᵢ | w₁, ..., wᵢ₋₁)
- 이전 단어들의 조건부 확률의 곱으로 문장 확률 계산

### N-gram 언어모델
- 마르코프 가정: 직전 n-1개 단어만 고려
- Unigram(N=1), Bigram(N=2), Trigram(N=3)
- 한계: 장거리 의존성 포착 불가, 데이터 희소성

### Bag of Words (BoW)
- 단어의 출현 빈도만 고려, 순서 무시
- 문서를 단어 빈도 벡터로 표현

### TF-IDF
- **TF(t,d)**: 문서 d에서 단어 t의 출현 빈도
- **IDF(t)**: log(N / (1+df)) — 많은 문서에 등장할수록 정보량 낮음
- **TF-IDF = TF × IDF**: 특정 문서에서 중요한 단어에 높은 가중치

### BM25
- TF-IDF의 확장, 문서 길이 정규화 포함
- 검색 엔진에서 문서 랭킹에 널리 사용

### 생성형 모델 평가 메트릭
- **비학습 기반**: BLEU, METEOR, TER, WER (어휘 기반)
- **비지도 학습 기반**: BERTScore, BARTScore (임베딩 기반)
- **지도 학습 기반**: BLEURT, COMET (학습된 회귀/랭킹 모델)

---

## Ch3. Pretrain 모델 기반 언어모델 연구

### 단어 임베딩
- One-hot → 저차원 밀집 벡터로 변환
- 의미적으로 유사한 단어가 벡터 공간에서 가까이 위치
- 예: king - man + woman ≈ queen

### Word2Vec
- **CBOW**: 주변 단어로 중심 단어 예측
- **Skip-gram**: 중심 단어로 주변 단어 예측
- 효율적인 학습으로 대규모 어휘에 적용 가능

### GloVe
- 공존 행렬(co-occurrence matrix) 기반
- 벡터 내적이 단어 쌍의 공존 확률 로그값과 일치하도록 학습
- 가중 함수로 빈도 높은 단어의 과대 가중 방지

### FastText
- Word2Vec의 Skip-gram 개선
- 서브워드(n-gram) 정보 활용 → OOV(미등록어) 처리 가능
- 예: "eating" → `<ea`, `eat`, `ati`, `tin`, `ing`, `ng>`

### Doc2Vec
- 문서 전체를 하나의 벡터로 표현
- 문서 ID 벡터를 단어 벡터와 함께 학습

### CoVe (Contextualized Vectors)
- 기계번역 모델의 인코더에서 문맥적 임베딩 추출
- GloVe + 번역 인코더 context vector → 다른 NLP 태스크에 적용

### 사전학습 모델 계보

#### ELMo
- Bidirectional LSTM 기반 사전학습
- 양방향 문맥 정보를 반영한 word representation

#### GPT
- Transformer Decoder-only 구조
- Causal Language Modeling (CLM): 왼쪽→오른쪽 단방향 예측

#### BERT
- Transformer Encoder-only 구조
- **MLM**: 토큰의 15%를 마스킹하고 예측
- **NSP**: 두 문장이 연속인지 판별
- Token + Segment + Position Embedding

#### XLNet
- Permutation Language Modeling (PLM)
- 다양한 단어 순서로 양방향 컨텍스트를 효율적으로 학습

#### RoBERTa
- BERT의 학습 전략 최적화
- Dynamic Masking, NSP 제거, 더 많은 데이터/배치/학습

#### ELECTRA
- Generator(BERT)가 [MASK]를 채우고, Discriminator가 원본/대체 판별
- 모든 토큰에서 학습 신호 → 더 효율적

#### ALBERT
- Cross-layer Parameter Sharing으로 파라미터 절감
- NSP → SOP (Sentence Order Prediction)

#### SpanBERT
- 연속된 span을 마스킹 (Span Masking)
- Span Boundary Objective: 경계 토큰으로 내부 예측
- QA, Coreference Resolution에서 우수

#### DistilBERT
- Knowledge Distillation으로 BERT 경량화
- Triple Loss: Distillation + Training + Cosine Loss
- BERT 대비 약 60% 크기, 97% 성능 유지

#### DeBERTa
- Disentangled Attention: content와 position을 분리하여 attention 계산
- Enhanced Mask Decoder: 마지막 레이어에서 absolute position 추가

#### XLM
- 다국어 사전학습 (MLM + Translation LM)
- Language Embedding 추가

#### BART
- Encoder-Decoder 구조의 denoising 사전학습
- 5가지 corruption: Token Masking/Deletion, Text Infilling, Sentence Permutation, Document Rotation

#### MASS
- Masked Sequence to Sequence Pre-training
- 인코더에 마스킹된 문장, 디코더에서 마스킹 부분만 예측

#### T5
- 모든 NLP 태스크를 Text-to-Text 형식으로 통일
- 입력에 태스크 prefix 추가 (예: "translate English to German:")

### 모델 경량화 기법
- **Quantization**: Float → Integer (QAT, PTQ)
- **Pruning**: 불필요한 가중치/뉴런/레이어 제거
- **Knowledge Distillation**: 큰 모델(Teacher) → 작은 모델(Student)
- **Low-rank Decomposition**: 가중치 행렬을 작은 행렬의 곱으로 근사

### 사전학습 모델의 한계
- 대규모 데이터/컴퓨팅 필요 (높은 CO₂ 배출)
- 파인튜닝 시 Catastrophic Forgetting
- 데이터 편향 (성별, 인종 등의 stereotype)
- Hallucination: 사실과 다른 내용 생성

### 미래 방향
- 지속적 학습 (Continual Learning) 파이프라인
- RAG (Retrieval Augmented Generation)
- Chain-of-Thought 추론
- RLHF / DPO를 통한 Human Alignment
- 편향 완화 학습

---

## Ch4. Large Language Model 기초

### LLM의 핵심 요소 4가지
1. **Infra**: 대규모 클라우드, 슈퍼컴퓨팅, GPU 클러스터
2. **Backbone Model**: 강력한 사전학습 모델 (GPT, LLaMA 등)
3. **Tuning**: 효율적 파인튜닝 (LoRA, QLoRA 등)
4. **Data**: 고품질 대규모 데이터, Instruction 데이터

### Human Alignment
- LLM 학습 2단계: ① 사전학습 (비정형 대규모 데이터) → ② Instruction Tuning (태스크 데이터, RLHF)
- 사전학습으로 지식 습득 후, alignment로 안전하고 유용한 응답 유도

### Scaling Law
- 모델 크기, 데이터 크기, 컴퓨팅량 → 성능이 power-law 관계로 향상
- 세 요소를 균형있게 확장해야 최적 성능

### In-Context Learning (ICL)
- **Zero-shot**: 태스크 설명만 제공
- **One-shot**: 태스크 설명 + 예시 1개
- **Few-shot**: 태스크 설명 + 예시 여러 개
- 모델이 클수록 ICL 능력이 급격히 향상

### LLM의 주요 방향성

| 방향 | 핵심 내용 |
|------|---------|
| Data | 다양한 도메인 커버리지가 중요, 데이터 품질이 성능 좌우 |
| Size | 사전학습 모델의 크기가 근본적 역량 결정 |
| Multimodal | 텍스트 + 이미지 + 음성 등 다중 모달리티 통합 |
| Multilingual | BLOOM(59개 언어), PaLM 2 등 다국어 지원 |
| Synthetic Data | LLM으로 학습 데이터 생성/증강 (2030년까지 합성 데이터가 실제 데이터 초과 전망) |
| Domain Specialized | 일반 모델 + 도메인 데이터로 특화 모델 구축 |
| Evaluation | HELM, LLM-as-Evaluator, 다차원 평가 |
| Prompt Engineering | CoT, Self-Consistency, Toolformer 등 |
| Open Source | LLaMA, Vicuna, Alpaca 등 오픈소스 생태계 확장 |

---

## Ch5. Large Language Model 한판정리

### LLM 학습 3단계
1. **Pre-Training**: 대규모 비정형 데이터로 언어 지식 습득 (가장 비용 큼)
2. **SFT (Supervised Fine-Tuning)**: Instruction 데이터로 미세조정
3. **RLHF / DPO**: 인간 선호도 기반 alignment

### Instruction Tuning
- 자연어 지시사항을 이해하도록 미세조정
- 모델의 Zero-shot 일반화 능력 크게 향상
- FLAN: Instruction Tuning한 137B 모델이 GPT-3 175B의 few-shot 성능에 근접

### Alignment Tuning
- **RLHF**: 선호도 데이터 → Reward Model 학습 → PPO로 LM 최적화
- **DPO**: Reward Model 없이 선호도 데이터로 직접 LM 최적화 (더 단순)

### 데이터 품질의 중요성
- 사전학습 데이터의 도메인 다양성이 높을수록 다운스트림 성능 향상
- 특정 도메인 제거 시 관련 태스크 성능 급락
- Filtering/Deduplication 등 전처리가 핵심

### Parameter Efficient Fine-Tuning (PEFT)
- **LoRA**: 가중치 행렬에 저랭크 행렬 추가, 원본 고정
- **QLoRA**: 4bit 양자화 + LoRA → 단일 GPU에서 대형 모델 튜닝 가능
- **Adapter**: 각 레이어에 작은 어댑터 모듈 삽입
- **Prefix-Tuning**: 입력 앞에 학습 가능한 prefix 벡터 추가

### Domain Specialization
- **Fine-tuning 방식**: 도메인 데이터로 추가 학습
- **RAG 방식**: 외부 지식을 검색하여 프롬프트에 주입
- 예: Med-PaLM 2 — 의료 QA에서 전문의 수준 달성

### LLM 평가
- **HELM**: 다차원 종합 평가 (정확도, 공정성, 편향, 독성, 효율성 등)
- **LLM-as-Evaluator**: GPT-4 등으로 자동 평가 (G-Eval)
- **FLASK**: Fine-grained skill별 세분화 평가
- **Hallucination 평가**: HaluEval, Pinocchio 등 사실성 검증
- **Toxicity/Fairness**: 페르소나별 독성 분석, 평가 편향 검증
- **리더보드**: Open LLM Leaderboard, MTEB, HELM

### Prompt Engineering 기법
- **Chain-of-Thought (CoT)**: 단계적 추론 유도 → 수학/논리 문제에서 큰 성능 향상
- **Zero-shot CoT**: "Let's think step by step" 추가만으로 추론 유도
- **Least-to-Most**: 복잡한 문제를 하위 문제로 분해 → 순차적 해결
- **Auto-CoT**: 클러스터링 기반으로 자동 데모 구성
- **Self-Consistency**: 여러 추론 경로 생성 후 다수결 투표
- **Verify-and-Edit**: 외부 지식 검색으로 CoT 결과 검증/수정

### LLM 도구 생태계
- **LangChain**: LLM 기반 앱 개발 프레임워크 (프롬프트 체이닝, 메모리, 도구 연결)
- **Scikit-LLM**: Scikit-learn 스타일의 LLM 인터페이스

### Emergent Abilities
- 특정 모델 크기 이상에서 갑자기 나타나는 능력
- 수학 문제 풀이, Instruction Following, Calibration 등
- CoT, Instruction Tuning이 이 능력을 크게 증폭

---

## Ch6. Multilingual, Multimodal, Cross-Lingual LLM

### Multilingual 사전학습 모델

#### Encoder-only
- **mBERT**: 104개 언어로 MLM 학습, 언어 임베딩 없이도 cross-lingual 전이
- **XLM**: MLM + Translation LM, Language Embedding 추가

#### Encoder-Decoder
- **mT5**: T5의 다국어 버전, 101개 언어
- **mBART**: BART의 다국어 버전, denoising 사전학습 → 기계번역 파인튜닝

### Multilingual LLMs
- **BLOOM**: 176B 파라미터, 59개 언어, 오픈 액세스
- **PaLM / PaLM 2**: 다국어 이해 및 추론 강화
- **LLaMA 2**: Meta의 오픈소스 LLM
- **GPT-3.5 / GPT-4**: 다국어 지원 강력

### Cross-Lingual Transfer
- WECHSEL: 단일 언어 모델의 서브워드 임베딩을 타 언어로 효과적으로 초기화
- 소스 언어 모델의 non-embedding 가중치를 복사하여 타겟 언어 모델 구축

### Multimodal LLMs 발전
- 2022: Flamingo → 2023: BLIP-2, GPT-4, LLaVA, MiniGPT-4 등 폭발적 증가
- **Gemini**: 텍스트 + 이미지 + 음성 + 영상을 통합 처리하는 네이티브 멀티모달 모델
- 다양한 입력 모달리티를 하나의 Transformer에서 처리하고 다양한 디코더로 출력

---

## Ch7. OpenAI 정리

### OpenAI API 주요 기능
1. **Chat (Text Generation)**: 대화형 텍스트 생성
2. **Embeddings**: 텍스트 벡터화 (분류, 검색, 클러스터링)
3. **Analysis**: 대규모 텍스트 요약/분석/QA
4. **Fine-tuning**: 커스텀 데이터로 모델 미세조정

---

## 핵심 키워드 요약

```
언어모델 발전: 통계 → 신경망 → Transformer → LLM
표현 방식: One-hot → Word2Vec/GloVe → Contextual (ELMo/BERT) → LLM
학습 패러다임: Pre-train → Fine-tune → Prompt → RLHF/DPO
핵심 기술: Attention, Scaling Law, ICL, CoT, RLHF, RAG, PEFT
평가: 자동 메트릭 → LLM-as-Evaluator → 다차원/세분화 평가
방향: Multimodal, Multilingual, Domain Specialization, Open Source
```
