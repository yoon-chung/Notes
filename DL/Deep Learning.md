# 딥러닝
## 1. 기본개념
### 1.1. 딥러닝 발전 5단계
1. Rule based programming: 모든 연산 방법을 사람이 전부 고안
2. 전통 머신러닝: SW1.5방식, 특징값을 뽑는 방식은 사람이, 특징값들로 판별하는 로직은 기계가 스스로 고안
3. 딥러닝: SW2.0방식, 모든 연산을 기계가 고안 (이미지면 픽셀, 텍스트면 토큰화)
4. Pre-training & Fine-tuning: 기존의 문제점인 분류대상/태스크가 바뀔 때마다 다른 모델 필요

4.1. 이미지의 경우:
- step1: Pretraining: 많은 동물을 구별하는 일반적 특징을 익히게
- step2: Fine-tuning: 태스크를 수행하기 위해 매핑 쪽에 해당하는 연산들만 새로 학습(판다, 토끼)

4.2. 텍스트 데이터 관점: GPT1
- step1: Pretraining: 사람의 개입없이 즉, 라벨링 없이 입력 텍스트에서 정답을 만들어낸다! (Un-supervised or Self-supervised pre-training)
- 예) task: 다음 단어 맞추기 => 학습 데이터 만들기: 데이터 쌍 만들기(1.곰세마리가, 한) (2.곰 세마리가 한, 집에) (3.곰세마리가 한 집에, 있어) 
- step2: Fine-tuning/ Transfer Learning: 예) task 텍스트 분류 긍정적?부정적? => 사람이 라벨링한 features들을 모아서 학습. frozen features

5. In-context learning (GPT3):
- 태스크 별 별도 모델이나 맞는 데이터 모을 필요없음. Pretraining된 모델로 여러 태스크 대응가능
- fine-tuning없이 태스크 설명 포함해서 text를 입력하면 output 출력함, pretraining된 모델만으로 fine-tuning만큼의 성능 얻음
- zero-shot: 태스크만 설명만, one-shot: 예제 하나, few-shot: 예제 여러개 추가
 <img src="images/dl1.png" width="800">


### 1.2. 딥러닝 기술 종류
- AI 구분법: 데이터: 정형데이터, 이미지/동영상, 텍스트, 음성, 학습: 교사, 비교사, 강화, 태스크: 인식, 생성


 
