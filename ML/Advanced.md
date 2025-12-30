# 개념정리
## 1. 프로세스
1. 데이터수집
2. 데이터전처리 - EDA, 결측치/이상치 처리, 연속형/범주형 변수 처리, 파생변수 처리, 변수 선택
3. 모델 선택 - 목적에 맞는 모델 선택, Hyperparameter 튜닝
4. 학습 및 평가 - 학습, 검증, 평가 dataset 분할, 모델 성능 측정
5. 배포
- 심화ML모델: Boosting 기반 - **Light GBM, CatBoost, XGBoost**

## 2. 데이터 전처리
### 2.1. 결측치/이상치 처리
1. 결측치
- 완전 무작위: 다른 변수와 관련 없음. 예) 임의누락
- 무작위: 다른 변수와 연관 있음. 예측 모델로 대체 가능
- 비무작위: 자기자신 변수와 연관. 예) 몸무게가 높은 응답군에서 몸무게 결측 발생 (숨기고 싶어서)
- 처리방법: 결측치 비율 10%미만: 제거/대체, 10~20%: 회귀 대체, Hot Deck, 20% 이상: 회귀 대체

2. 이상치
- 탐지: Z-score(값이 2.5 ~ 3정도면 이상치로 판별), IQR
- 처리방법: 삭제, 대체: 통계치(평균, 중앙, 최빈), 상/하한값 정하고 경계를 넘어갈때 대체, knn 등 모델로 이상치 예측가능, 변환(Transformation): log변환, 제곱근 변환

### 2.2. 연속형/범주형 변수 처리
#### 1. 연속형 변수 변환 
0. 구조
```shell
1. 함수변환 - 로그/제곱근/Box-Cox
2. 스케일링 - 정규화/표준화/로버스트
3. 구조화
```
1. 함수 변환
- **로그**: 비대칭된 분포를 정규분포에 가깝게 전환, 데이터 정규성은 모델 성능 향상, 스케일을 작게 만들어 데이터간 편차 줄임, 이상치 완화, 선형관계 생성, 단, 0이나 음수 적용은 불가
- 제곱근: 로그 변환과 비슷, 비대칭 강도가 강하면 로그변환, 강도가 약하면 제곱근변환 유리 (로그보다 줄어드는 변환 폭이 작음), 
  - 로그/제곱근 : 왜도(Skewness)를 줄여 비대칭(Right&Left-Skewed)된 분포를 정규분포에 가깝게 전환하도록 도움 -> 모델 성능 향상, 데이터간 편차를 줄임. 정규분포에 가깝게 변환. 
다만, 오른쪽으로 치우쳐진 데이터 분포에 활용시 더 심하게 오른쪽으로 치우치는 부작용. 큰 값을 작게 만들기 때문에 큰 값의 분포가 많아지기 때문. 분포 먼저 확인 필요.
- Box-Cox: 분포가 치우친 방향과 무관하게 변환 가능. 람다 파라미터를 조절하면서 유연하게 분포를 변화. 사용자가 정하는 주관적 파라미터이므로 최적 람다 값 찾아야.
2. 스케일링: 연속형 독립변수들의 수치범위, 즉 영향력을 동등하게 변환 -> 경사하강법, KNN등 거리기반 알고리즘에 효과적. 예) 키 170cm, 몸무게 60kg
- **정규화(Min-Max)**: 0-1 사이 전환, 단 이상값에 취약
- **표준화**: 평균 0, 표준편차 1 (Z-score). 평균과의 거리를 표준편차로 나눔.
- 로버스트: 중앙값 활용
3. 구간화
- 수치형->범주형 변수로 전환시키는 방법. 예) 10 ~ 19 -> 10대, 20~29 -> 20대
- 데이터가 범주화되므로, 학습 모델의 복잡도 감소
- 등간격(동일 간격), 등빈도(동일 갯수)

#### 2. 범주형 변수 변환
1. 원핫 인코딩: 0, 1로 구성된 binary 벡터 형태, 메모리 usage 문제
2. 레이블 인코딩: 각 범주를 정수로 표현, 순서 있는 변수들에 효율적
=> 순서가 존재하지 않는 변수에 적용하면 알고리즘 오해석 가능성. (변수간 상대적 크기를 비교하는 알고리즘)  
3. 빈도 인코딩: 고유 범주의 빈도 값을 인코딩. 특정 관측치의 빈도가 높을수록 높은 정수값을, 빈도가 낮을수록 낮은 정수 값을 부여.
=> 다른 특성의 의미를 지니고 있어도 빈도가 같으면, 다른 범주간 의미가 동일하게 인식될 수 있는 가능성.
=> 레이블+빈도 인코딩 결합하여 활용
4. 타겟 인코딩: 통계량(평균)으로 인코딩. 파생변수와 비슷. 과적합/data-leakage 가능성 있음.
=> 과적합 방지: smoothing(전체 dataset 평균에 가깝게 전환), k-fold 
5. 인코딩 패키지
- scikit learn
```shell
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import TargetEncoder
```
- **category_encoders**
```shell
import category_encoders as ce
encoder = ce.BackwardDifferenceEncoder(cols=[...])
encoder = ce.BaseNEncoder(cols=[...])
encoder = ce.BinaryEncoder(cols=[...])
encoder = ce.CatBoostEncoder(cols=[...])
encoder = ce.CountEncoder(cols=[...])
encoder = ce.GLMMEncoder(cols=[...])
encoder = ce.GrayEncoder(cols=[...])
encoder = ce.HashingEncoder(cols=[...])
encoder = ce.HelmertEncoder(cols=[...])
encoder = ce.JamesSteinEncoder(cols=[...])
encoder = ce.LeaveOneOutEncoder(cols=[...])
encoder = ce.MEstimateEncoder(cols=[...])
encoder = ce.OneHotEncoder(cols=[...])
encoder = ce.OrdinalEncoder(cols=[...])
encoder = ce.PolynomialEncoder(cols=[...])
encoder = ce.QuantileEncoder(cols=[...])
encoder = ce.RankHotEncoder(cols=[...])
encoder = ce.SumEncoder(cols=[...])
encoder = ce.TargetEncoder(cols=[...])
encoder = ce.WOEEncoder(cols=[...])
```

### 2.3. 파생변수 생성 (Feature Engineering)
기존 변수 정보를 토대로 정제 및 생성된 새로운 변수

#### 1. 기존 변수 정제 : 함수변환, 스케일링, 구간화, 기타
기타: 
- 분할: 명량(2014) -> 영화: 명량 , 개봉연도: 2014
- 시간변수: 날짜 -> 년, 월, 주차, 일년기준 지난일수(1~365), 시간, 시간대(새벽, 늦은아침..)
- 시간차: 거래일-첫 거래일

#### 2. 변수들 간 관계 활용 : 상호작용, 통계기반
도메인 지식과 연관. 예) 2층 주택 넓이 = 1층 + 2층 넓이
- 상호작용: 변수 결합. 예) 통신사 월 이용 금액을 활용해 멤버십 등급을 파생
- 통계 기반: 평균, 중앙, 최소, 최대 등 활용. 예) 상품의 가격대(비싼정도) = 가격/카테고리별 평균값

### 2.4. Feature Selection 변수선택
1. Filter methods: 통계적 관계. 예) 상관관계 높은 변수들 제거, 분산 낮은 변수들 제거. 
- 카이제곱 test기반: Y-X 카이제곱 통계량 구하고 p-value가 충분히 낮으면 Y와 해당 독립변수 X가 관계가 있다고 판단하여 변수로 선택한다. 반대라면, 변수 제거.
2. Wrapper methods: 실제 ML모델 성능을 활용해 변수 선택. 예) Forward selection, Backward elimination
- 장점: 모델 성능에 따른 최적의 변수 집합 도출
- 단점: 반복 학습에 따른 시간, 비용
3. Embedded methods: 모델 훈련 과정에서 중요도 설정, 변수 선택. 예) Feature importance, Regularizer 기반 선택
- 장점: 변수 중요도 & 모델 복잡성 동시 고려
- 단점: 훈련과정에서 선택된 변수 조합 해석의 어려움
- Feature importance :트리node분할(Gini계수, Entropy활용). 특정 feature가 트리의 순수도가 높은 분할에 도움이 된다면 해당 feature는 중요도, 즉 모델 성능향상에 기여도가 높다고 이해. 
- Regularization 기반: 특정 feature의 weight를 0 or 0에 가깝게 만들어 feature 제거 효과. L1: weight를 0으로 변환. L2 & ElasticNet: weight를 0에 가깝게 변환
4. 실전 사례: 
- 상관계수가 높은 하위 집합에서, 가장 원소의 개수가 많은 변수를 대표로 선택하고 나머지 삭제.
- 학습된 트리모델의 중요도 낮은 변수들 제거: feature importance 낮으면 적은 기여할것이라는 가정
- permutation importance: 검증 데이터셋의 특정 변수들을 무작위로 permutation 이후 성능산출. 원본 데이터셋보다 성능 떨어지면 그 변수는 중요한 변수. 검증 데이터셋의 Feature를 하나하나 shuffle하며 성능 변화를 관찰. 만약 해당 Feature가 중요한 역할을 하고있었다면, 모델 성능이 크게 하락할 것. 
- target permutation : Shuffle된 Target 변수 모델을 학습시킨 후, Feature importance와 Actual feature importance를 비교해 변수를 선택. Shuffle의 의미- 해당 feature를 noise(의미없는 변수)로 만드는 과정
- adversarial validation: train set과 valid set이 얼마나 유사한지 판단. 이 모델 분류에 도움이 되는 feature는 과적합 유발할 수 있으니 제거.

## 3. 모델 선택
### 3.1. 선형회귀
1. 최소자승법 OLS: SSE를 최소화하는 회귀계수 Beta들을 구하는 방법
- 장점: 학습/예측 속도 빠름. 해석이 명확
- 단점: 현실에서는 잘 적용되지 않을 가능성. 이상치에 민감
2. 가정 4가지
- 선형성: x, y 사이 선형관계 -> 비선형성 해결방법: 해당변수 제거, 로그/제곱근 변환하여 비선형성 완화
- 정규성: 잔차들의 평균이 0인 정규분포
- 등분산성: 잔차들의 분산이 일정
- Q-Q plot: 잔차관련 가정 확인 -> 해결방법: 로그/루트 취하기, 더 많은 데이터를 수집
- 독립성: x들간 상관관계 존재하지 않아야함 -> 아닐경우 변수 제거, 상관관계표를 그려서 제거
- VIF: 다중공선성 확인방법, VIF > 10이면 문제있음 -> 다중공선성 높은 변수 조합 중 하나의 변수를 제거
3. R-square 결정계수: x들이 설명할 수 있는 y의 변동
4. 실전사례: 최신 LLM데이터의 평가모델로 활용 : 평가지표를 variable로 두고 quality를 예측하는 모델로 활용

### 3.2. KNN
1. 거리기반, 사례 기반 학습. 가까운 이웃에 위치한 k개 데이터를 고려
- 유사한 데이터(인접 이웃)을 위한 방법론으로도 사용
- 추천시스템 모델의 이웃정보를 side information으로 활용
- 결측 보간에도 활용: 나이 추정에 인접한 이웃데이터 활용
2. 장단점
- 장점: 단순, 빠름, 특별한 가정이 존재하지 않음
- 단점: 스케일링 필수, 적절한 k 선택 필요(cross validation통해 적절한 k개 찾음)

### 3.3. Decision Tree
1. 데이터내 규칙 을 찾아 데이터를 구분, 맨 마지막 리프노드의 불순도를 최소화
- 루트노드, 리프노드
- 불순도가 높으면 지니계수, 엔트로피가 높게 나타남 -> 이 지표들을 줄어드는 방향으로 진행
- 불순도가 가장 낮은 값을 기준으로 먼저 분할 진행
- 순도가 100% 되는 지점에서 분할 종료
2. 장단점
- 장점: 직관적, 스케일에 민감하지 않음. 범주/연속형 모두 처리 가능
- 단점: 트리가 깊어지면 과적합 위험-> 적절한 pruning필요. 새로운 샘플에 취약
3. Pruning(가지치기)
- 트리의 Depth, Leaf 적절히 조정하는 Hyperparameter 존재
4. Tree 시각화: 결정경계와 트리 분할 과정을 시각화 가능
5. Feature importance: 불순도를 많이 낮출 수 있는 feature = 중요 feature

### 3.4. 앙상블
0. 앙상블 히스토리: Decision Trees - Bagging - Random Forest - Boosting - Gradient Boosting - XGBoost - Light GBM - CatBoost
1. 여러개 서로 다른 예측 모형을 생성해 결과를 종합, voting(범주형), average(연속형)
2. 트리앙상블: 배깅, 부스팅 
3. 부트스트랩: 복원추출로 모집단의 통계량을 추론, 실제 모집단 통계량의 분포와 비슷
4. 배깅: 부트스트랩의 복원추출 과정을 ML앙상블에 활용. 표본을 여러번 뽑아 모델 학습하고 결과를 집계. 장점: 오버피팅 가능성 감소, 일반화 성능 증가, 분산 감소
5. 배깅 vs 부스팅:
  - 배깅: 전체에서 학습 데이터를 무작위 추출. 각 모델이 독립적으로 구축
  - 부스팅: 이전 모델에서 잘못 에측된 집합을 다음 모델 학습 데이터로 추출. 이전 모델 성능에 영향을 받아 편향을 감소하는데 집중

#### 3.4.1. Random Forest
1. 배깅의 일종. 독립변수 x도 무작위로 선택해 트리를 생성
- N개의 Tree에서 발생한 output을 voting(범주형, 분류)/average(연속형, 회귀)하여 최종 output생성 -> High Var, Low Bias 상황에서 분산 감소에 도움
2. 장단점
- 장점: 예측 성능 향상, 과적합 완화
- 단점: 대용량 데이터 학습 오랜시간 소요, 단일 트리모델보다 해석력이 떨어짐.

#### 3.4.2. GBM
0. 참고 >> AdaBoost: 이전 모델이 틀리게 예측한 값에 가중치 부여, 다음 모델 조정에 활용.
1. GBM : 정답과 이전 모델의 예측값으로 Gradient를 계산하여 다음 모델을 조정.
2. Gradient Descent: 오차함수(Loss Function)의 Loss값이 최소가 되는 지점까지 반복. 









