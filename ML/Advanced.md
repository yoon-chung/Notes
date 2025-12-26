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
- **표준화**: 평균 0, 표준편차 1 (Z-score)
- 로버스트: 중앙값 활용
3. 구간화
- 수치형->범주형으로 전환시키는 방법. 예) 10 ~ 19 -> 10대, 20~29 -> 20대
- 등간격(동일 간격), 등빈도(동일 갯수)

#### 2. 범주형 변수 변환
1. 원핫 인코딩: 0, 1로 구성된 binary 벡터 형태, 메모리 usage 문제
2. 레이블 인코딩: 각 범주를 정수로 표현, 순서 있는 변수들에 효율적
3. 빈도 인코딩: 고유 범주의 빈도 값을 인코딩 
=> 레이블+빈도 인코딩 결합하여 활용
4. 타겟 인코딩: 통계량(평균)으로 인코딩
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







