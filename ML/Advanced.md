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






