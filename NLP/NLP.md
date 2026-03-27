# NLP (자연어 처리)
- 일상생활언어, 인간의 언어 (반대말: 인공언어, 프로그래밍언어)
- 컴퓨터가 자연언어의 의미를 분석, 이해, 생성할 수 있게 만들어주는 기술
- NLG(생성), NLU(이해) -> 교집합 사례: 챗봇
- 한국어 어려움 (어근, 접사. 주어생략, 어순, 띄어쓰기)

## 1. 언어학
- 음절: 가장 작은 말소리의 단위 (한국어에서는 보통 한글자)
- 형태소: 언어에서 의미 갖는 가장 작은 단위
- 어절: 띄어쓰기 단위
- 품사: 문법상 역할/의미/형태에 따라 구분
### 1. **태깅**: 품사를 태깅. 
- 형태론(morphology) - 의미갖는 기본단위인 형태소를 분석하는 학문 
- 이형태: 한 형태소에 대해 여러개 변이형태 갖는다. (walk -> walked, go -> went)
### 2. **파싱**: 트리를 만듬. 
- 통사론(syntax) - 단어가 결합하여 구/문장 형성하는 규칙/방법 연구하는 학문 
- 구조적 모호성, 반복 
### 3. 의미론 (semantics)
- 구문 구조는 정상이지만 의미적으로 어색한 문장
- 의미역(semantic roles): 문장에서 각 단어의 의미적 역할 분석. 행위자/대상자
- 동의어/반의어, 상하관계, 동음이의어, 다의어, 연어(collocation)  
### 4. 화용론
- 보이지 않는 의미. 화자가 의미하는 바에 대한 연구
- 문맥, 직시표현(물리적문맥 없으면 이해안됨), 지시, 추론, 대용어(이미 소개된 실체에 뒤따르는 지시), 전제(진리, 사실), 화행(발화와 함께 취해지는 행위)
### 5. 담화론
- 대화/여러문장 연구하는 학문
- 결속(의미적 연결성 분석), 일관성(대화에 지속참여하며 일관성 발견), 차례얻기(접속사로 말을 이어나감), 협조의 원칙, 함의(화자의 암시)
### 6. 자연어처리의 언어학
- 키워드 분석, 토큰화(전처리의 첫단계), 품사태깅(POS Tagging), 구문분석, 의미/담화분석, 문법 교정, BERT(풍부한 언어정보 계층구조를 반영하는 모델)

## 2. 텍스트 전처리
### 1. 전처리 (Preprocessing)
- data cleaning, normalization, feature selection, transformation, missing values imputation, instance selection, data integration, noise identification, discretization
- HTML태그, 특수문자, 이모티콘, 정규표현식, stopword(불용어), stemming(어간추출), lemmatizing(표제어추출)
- 파이프라인: 문서->토큰화->텍스트전처리->Bag of Words
- KoNLPy: 한국어 자연어처리 위한 파이썬 lib
- NLTK: 영어 텍스트 위한 파이썬 lib

### 2. 토큰화 (Tokenization)
- 토큰 기준 다름 (어절, 단어, 형태소, 음절, 자소)
- 고려사항: 구두점, 특수문자, 줄임말과 단어내 띄어쓰기, 문장 토큰화
- 한국어 토큰화의 어려움. 형태소 단위의 토큰화가 필요하기 때문
- KoNLPy: morphs(형태소 추출), pos(품사 태깅), nouns(명사 추출)
- SentencePiece: 구글의 토큰화 도구

### 3. 정제 (Cleaning)
- stopwords: NLTK에서는 여러 불용어를 사전 정의해둠. (아, 휴, 음, 어..) 

### 4. 정규화 (Normalization)
- stemming vs lemmatization: 어간 추출 vs 원형 복원. 심플신속 vs 정확느림
- stemming: allowance -> allow, formalize -> formal 
- lemmatization(표제어 추출): watched -> watch, has -> have, cats -> cat 

### 5. 편집거리 (Edit distance)
- Levenshtein distance: 한 string s1을 s2로 변환하는 최소 횟수를 거리로 표현
- string을 변화하기 위한 edit 방법을 세가지로 분류

### 6. 정규표현식 
- Regex: 특정한 규칙 가진 문자열 집합을 표현하는데 사용하는 형식 언어 

```
[ ]   : 문자, 숫자 범위를 표현하며 +, -, . 등의 기호를 포함
{개수}   : 특정 개수의 문자, 숫자를 표현
{시작개수, 끝개수}   : 특정 개수 범위의 문자, 숫자를 표현
+   : 1개 이상의 문자를 표현. 예) a+b 는 ab, aab, aaab
*   : 0개 이상의 문자를 표현. 예) a*b 는 b, ab, aab, aaab
?   : 0개 또는 1개의 문자를 표현. 예) a?b 는 b, ab
.   : 문자 1개만 표현
^   :  [ ] 앞에 붙이면 특정 문자 범위로 시작하는지 판단
/   :  [ ] 안에 넣으면 특정 문자 범위를 제외
$   : 특정 문자 범위로 끝나는지 판단
|   : 여러 문자열 중 하나라도 포함되는지 판단
( )   : 정규표현식을 그룹으로 묶음. 그룹에 이름을 지을 때는 ?P<이름> 형식
\   : 정규표현식에서 사용하는 문자를 그대로 표현하려면 앞에 \를 붙임. 예) \+, \*
\d   : [0-9]와 같음. 모든 숫자
\D   : [^0-9]와 같음. 숫자를 제외한 모든 문자
\w   : [a-zA-Z0-9_]와 같음. 영문 대소문자, 숫자, 밑줄 문자
\W   : [^a-zA-Z0-9_]와 같음. 영문 대소문자, 숫자, 밑줄 문자를 제외한 모든 문자
\s   : [ \t\n\r\f\v]와 같음. 공백(스페이스), \t, \n, \r, \f, \v을 포함
\S   : [^ \t\n\r\f\v]와 같음. 공백을 제외하고 \t, \n, \r, \f, \v만 포함
```

## 3. 자연어이해 하위분야
### 1. 형태소 분석기
- 규칙/통계/딥러닝 기반
- 품사 태깅
- HMM(Hidden Markov Model): 통계적 모델. 바로 직전 단계에서만 직접적인 영향을 받음 
- CRF(Conditional Random Field: 시퀀스 라벨링에 이용. 전체 문장의 맥락을 고려
### 2. 개체명 인식
- 특정 명사, 태깅 시스템: BIO시스템 주로 사용 (begin, inside, outside)
### 3. 정보 추출
- 구조적인 triple(주어-관계-목적어로 나타낸 구조)을 추출하는 태스크
- 정보추출 시스템 구조: 분할-토큰화-품사태깅-엔티티 추출-엔티티쌍의 특정패턴 추출
### 4. 텍스트 분류
- 분류 or 클러스터링
- 텍스트 분류 프로세스: 전처리-토큰화-feature extraction-학습-모델
- 예: 감성 분석(뉴스, 영화 리뷰), 스팸메일 필터링, 대화의도 분류, 상품 카테고리 분류, 혐오표현 분류

## 4. 자연어생성 하위분야
### 1. 기계번역
- 규칙 기반, 통계 기반, 신경망기반(인코더-디코더 구조)
### 2.  질의응답
- 딥러닝 기반. 정보검색+질의응답, 대화형(챗gpt)
### 3. 대화시스템
- 사용자 주도->시스템 주도
- 자연어 이해: 도메인 확인, 의도파악, 슬롯 채우기 
- 대화상태 추적: 대화 히스토리 정보 반영하여 대화상태 추적
- 일상 대화 시스템: 검색 기반 방식, 생성 기반 방식(기계번역의 Seq2Seq과 유사), 검색-생성 혼합 방식 
### 4. 문서 요약
- 추출 요약, 추상적 요약(다른 표현으로 재구성) 

## 5. 자연어처리 역사
### 1. 규칙/통계 기반: 전문가가 만듬
### 2.  ML/DL 기반
- 지도학습, 비지도학습
- 인공지능 > 기계학습 > 딥러닝
- 딥러닝: feature extraction까지도 수행. 때문에 데이터 품질+양 중요해짐
### 3.  뉴럴심볼릭 기반: 전문가 데이터 활용하여 딥러닝 모델에 주입
### 4.  Pretrain-Finetuning 기반
- 대량의 말뭉치로 언어 능력을 pre-training 이후 task-specific fine-tuning
- Pretraining: 내가 원하는 task 이외의 다른 task의 데이터를 이용하여 주어진 모델을 먼저 학습하는 과정
- Finetuning: 사전학습된 모델을 원하는 task에 해당하는 데이터, 학습 방식으로 다시한번 재학습 시키는 과정
### 5. LLM 기반
- Scaling Laws for Neural Language Models (OpenAI)
- Foundation Models: 만능 모델 (특정 task에 국한되지 않음)
- Prompt engineering과 같은 새로운 직군 등장

## 6. 딥러닝 기반 기초 (BERT 이전)
### 1. RNN
- sequence-to-sequence: 입력-출력. 기계번역(파파고)
- 인코더: 입력 시퀀스를 받아 고정된 길이의 벡터로 변환. 문맥 벡터
- 디코더: 문맥 벡터를 받아 출력 시퀀스를 순차생성. (auto regressive)
- RNN(Recurrent Neural Networks): 시계열특성 다루는데에 효과적. 때문에 자연어처리에 효과적임 (이미지에는 CNN적합) 
- 모든 파라미터를 공유. 시간순서역전파
- LSTM(Long short-term Memory): RNN에서 발생하는 Long-term dependency problem 완화 방법. 필요한 정보만을 선택적으로 업뎃/삭제 방법 도입 (정보를 잘 기억하고 활용)

### 2. Attention
- 문맥에 따라 집중할 단어를 결정하는 방식. Long-term dependancy해결
- 시퀀스의 길이가 길수록 attention이 없으면 성능 저하
- Seq2Seq RNN 구조 위에 Attention 구조를 추가한 형태. 디코더가 어디에 집중할지 반영하여 학습 고도화

### 3. Transformer
- "Attention is All you need" 논문에서 소개
- Self-Attention: 같은 문장 내 토큰들끼리 어텐션 취함. RNN 존재하지 않음
- Multi-head Attention: Attention(Query, Key, Value)을 여러 개 Head 로 나눠서 병렬로 계산
- Positional Encoding: 입력 시퀀스의 단어들이 어떤 순서로 들어왔는지 에 대한 정보가 누락되기에 이를 보완. Input embedding 값에 더하여 인코더와 디코더의 입력값으로 입력

## 7. BERT (Bidirectional Encoder Representations from Transformers)
### 1. 개요
- Google, 2018. "Pre-training of Deep Bidirectional Transformers for Language Understanding"
- Transformer의 **인코더**만 사용한 모델
- 사전학습(Pre-training) + 미세조정(Fine-tuning) 패러다임의 대표 모델
- 핵심: **양방향(Bidirectional)** 문맥 이해. 왼쪽+오른쪽 문맥을 동시 반영

### 2. 사전학습 (Pre-training)
- 대량의 비라벨 텍스트 데이터로 언어 자체를 학습
- **MLM (Masked Language Model)**
  - 입력 토큰의 15%를 [MASK]로 치환 후 해당 토큰을 예측
  - 80% [MASK] 토큰으로 대체 / 10% 랜덤 토큰으로 대체 / 10% 원래 토큰 유지
  - 양방향 문맥을 학습할 수 있는 핵심 전략
- **NSP (Next Sentence Prediction)**
  - 두 문장이 연속인지 아닌지 이진 분류. 문장 간 관계 이해 학습 (IsNext / NotNext)

### 3. 입력 표현 (Input Representation)
- 세 가지 임베딩의 합:
  - **Token Embedding**: WordPiece 토큰화 (서브워드 단위)
  - **Segment Embedding**: 문장 A / 문장 B 구분
  - **Position Embedding**: 토큰의 위치 정보 (학습 가능한 파라미터)
- 특수 토큰: [CLS] 문장 시작(분류용), [SEP] 문장 구분

### 4. 미세조정 (Fine-tuning)
- 사전학습된 BERT 위에 task-specific layer 추가 후 재학습
- 적용 태스크 예시:
  - 문장 분류: [CLS] 토큰의 출력 벡터 활용 (감성분석, 스팸분류 등)
  - 개체명 인식(NER): 각 토큰의 출력 벡터로 BIO 태깅
  - 질의응답(QA): 정답 시작/끝 위치를 예측 (SQuAD)
  - 문장 쌍 분류: 두 문장 관계 판별 (유사도, NLI)

### 5. 모델 구조
- BERT-Base: 12 layers, 768 hidden, 12 heads, 110M 파라미터
- BERT-Large: 24 layers, 1024 hidden, 16 heads, 340M 파라미터

### 6. 한국어 BERT
- KoBERT (SKT), KorBERT (ETRI), multilingual BERT (Google)
- 한국어 특성상 형태소 단위 토큰화가 성능에 큰 영향

---

## 8. GPT (Generative Pre-trained Transformer)
### 1. 개요
- OpenAI. GPT-1(2018), GPT-2(2019), GPT-3(2020), GPT-4(2023)
- Transformer의 **디코더**만 사용한 모델
- 핵심: **단방향(Unidirectional/Auto-regressive)** 텍스트 생성. 왼쪽→오른쪽 순차 예측
- BERT와의 차이: BERT는 이해(인코더), GPT는 생성(디코더) 중심

### 2. 사전학습 방식
- **CLM (Causal Language Modeling)**
  - 이전 토큰들을 보고 다음 토큰을 예측 (auto-regressive)
  - P(w_t | w_1, w_2, ..., w_{t-1})
  - Masked Self-Attention: 미래 토큰을 볼 수 없도록 마스킹

### 3. GPT 시리즈 발전
- **GPT-1**: 사전학습 + 미세조정 구조 제시. 12 layers
- **GPT-2**: 미세조정 없이도 다양한 태스크 수행 가능 (Zero-shot). 1.5B 파라미터
- **GPT-3**: Few-shot / In-context Learning의 등장. 175B 파라미터
  - In-context Learning: 별도 학습 없이 프롬프트에 예시를 넣어 태스크 수행
  - Zero-shot: 예시 없이 지시만으로 수행
  - One-shot: 예시 1개 제공
  - Few-shot: 예시 여러 개 제공
- **GPT-4**: 멀티모달(텍스트+이미지 입력), 더 긴 문맥 처리

### 4. RLHF (Reinforcement Learning from Human Feedback)
- ChatGPT(GPT-3.5)에 적용된 학습 방식
- 과정: SFT(Supervised Fine-Tuning) → Reward Model 학습 → PPO로 강화학습
- 인간의 선호도를 반영하여 더 유용하고 안전한 응답 생성

### 5. Scaling Law
- 모델 크기(파라미터), 데이터 양, 연산량이 증가할수록 성능이 예측 가능하게 향상
- Emergent Abilities: 일정 규모 이상에서 갑자기 나타나는 능력 (추론, 코드생성 등)

---

## 9. BART (Bidirectional and Auto-Regressive Transformers)
### 1. 개요
- Facebook(Meta), 2019. "Denoising Sequence-to-Sequence Pre-training"
- Transformer의 **인코더 + 디코더** 모두 사용 (Seq2Seq 구조)
- BERT(양방향 인코더) + GPT(자기회귀 디코더)의 장점 결합
- 핵심: **노이즈 제거(Denoising)** 방식의 사전학습

### 2. 사전학습 (Noising + Denoising)
- 원본 텍스트에 노이즈를 추가한 뒤, 원본을 복원하도록 학습
- **노이즈 기법 (Corruption Schemes)**:
  - Token Masking: BERT처럼 토큰을 [MASK]로 치환
  - Token Deletion: 토큰을 삭제 (위치 정보도 사라짐 → 더 어려운 태스크)
  - Text Infilling: 연속된 토큰 span을 하나의 [MASK]로 대체 (span 길이는 포아송 분포)
  - Sentence Permutation: 문장 순서를 랜덤으로 섞음
  - Document Rotation: 문서의 시작점을 랜덤으로 변경
- Text Infilling이 가장 효과적인 것으로 실험 결과 확인

### 3. 모델 구조
- 인코더: 양방향. 노이즈가 추가된 입력을 처리 (BERT와 유사)
- 디코더: 자기회귀. 원본 텍스트를 순차적으로 복원 (GPT와 유사)
- BERT-Large와 동일한 규모: 각각 6 layers 인코더 + 6 layers 디코더

### 4. Finetuning 및 적용 태스크
- **문서 요약**: BART가 특히 강점을 보이는 태스크 
- **기계번역**: 인코더를 다른 언어의 인코더로 교체하여 활용 가능
- **텍스트 분류**: 디코더의 마지막 토큰 출력을 분류에 활용
- **질의응답**: 인코더에 질문+문서, 디코더에서 답변 생성
- **추상적 요약(Abstractive Summarization)**: Seq2Seq 구조의 장점이 발휘됨

### 5. BERT / GPT / BART 비교

| 구분 | BERT | GPT | BART |
|------|------|-----|------|
| 구조 | 인코더 | 디코더 | 인코더+디코더 |
| 방향 | 양방향 | 단방향(→) | 양방향 인코더 + 단방향 디코더 |
| 사전학습 | MLM, NSP | CLM | Denoising (노이즈 복원) |
| 강점 | 이해(분류, NER, QA) | 생성(텍스트 생성) | 이해+생성(요약, 번역) |
| 대표 활용 | 감성분석, 개체명인식 | 챗봇, 코드생성 | 문서요약, 기계번역 |




