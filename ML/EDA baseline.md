# EDA baseline
## 1. 환경설정
```shell
# 계산, 데이터셋 로드, 정제
import numpy as np
import pandas as pd
import os

# 전처리
from scipy.special import boxcox1p
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from category_encoders import CountEncoder

# 출력 및 시각화
import pprint
import matplotlib.pyplot as plt
import matplotlib.dates as mpdates
import seaborn as sns

from IPython.display import display
from mplfinance.original_flavor import candlestick_ohlc

# 셀 실행 후 경고 무시
import warnings
warnings.filterwarnings(action='ignore')
```

## 2. DF 만들기

- df = pd.read_csv('{PATH / FILE NAME}', encoding='{utf-8, cp949}')


## 3. DF 확인
```shell
# 컬럼 전체를 리스트로 반환
df.columns

# 행 인덱스 정보 제공 
df.index 
df1 = df[df['col'] == 'NIKE'].copy() # col내용이 'NIKE'인 컬럼정보만 선택, 복사
df['new col name'] = df.index # 설정된 인덱스를 하나의 컬럼으로 추가
# 행 인덱스 날짜로 되어있다면, Date컬럼을 새로 만들고 인덱스 초기화
df['Date'] = df.index
df.reset_index(drop=True, inplace=True)

# 행, 열의 크기
df.shape

# 컬럼 정보와 데이터 타입, 총 행의 갯수 요약하여 보기 
df.info()

# 컬럼별 기초 통계량 확인
df.describe()

# 중복 제거하고 리스트로 반환
df['col'].unique()

# 중복 제거하고 갯수 반환
df['col'].value_counts() 

# 지정한 컬럼의 데이터에 대하여 행을 슬라이싱하여 데이터 반환  
df['col'][:]

# 지정한 행번호와 열번호(슬라이싱 가능)의 데이터를 반환
df.iloc[row no, col no]

# 지정한 컬럼들을 기준으로 행번호(슬라이싱 가능)에 해당하는 데이터를 반환
df.loc[row no, ['col1 name', 'col2 name', ]] 
df = df.loc["2025-01-01":"2025-03-01"]

# 슬라이싱 한 구간만큼의 컬럼명들을 리스트로 반환
df.columns[:]

# 불필요한 컬럼 제거
df.drop([col1, col2,..,col3], axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)

# 날짜-> 숫자
df['Date'] = df['Date'].map(mpdates.date2num)

# 숫자 -> 날짜
ax.xaxis.set_major_formatter(mpdates.DateFormatter('%Y-%m-%d')) # 시각화
```

## 4. 전처리 

### 4-1. 결측치 확인
```shell
df.isnull(), df.isna()

# null이 하나라도 있는 행을 불리안 인덱싱 반환
df.isnull().sum(axis=1) > 0 # axis = 1(행) 
df.isna().sum(axis=1) > 0 # axis = 1(행)

# 특정컬럼에서 몇개 열이 결측치 영향 받는
display(df[df['col'].isna() == True])

# 범주형 변수와 수치형 변수로 나눠서 결측치를 확인
missing_cat_cols = df.loc[:, df.isna().sum() > 0].select_dtypes("object").columns
missing_num_cols = df.loc[:, df.isna().sum() > 0].select_dtypes(np.number).columns
display("범주형 변수 결측치 현황")
display(df.loc[:, missing_cat_cols].isna().sum())
display("수치형 변수 결측치 현황")
display(df.loc[:, missing_num_cols].isna().sum())
# 범주형 변수의 결측치 시각화
missing_cat_df = pd.DataFrame(df.loc[:, missing_cat_cols].isna().sum())
sns.barplot(data=missing_cat_df, x=missing_cat_df.index, y=missing_cat_df[0])
```

### 4-2. 결측치 제거

- df_copy = df : 원본 데이터 보관 후 작업

- indices_to_keep = df.isnull().sum(axis=1) < 10 : 결측치 제거 대상을 제외하는 선정 

- df = df[indices_to_keep].copy() : 결측치를 제거하고 데이터 복사

- df['col name'].fillna(0, inplace=True) : na를 0으로 채워줌

- df.dropna(axis=0, inplace=True) : axis=0일 경우 결측치가 있는 행을, axis=1이면 열을 전부 제거 #.sum()일때는 반대(axis=0: 열, axis=1: 행 합계)
- df = df[(df[[col1, col2, col3,,,]] != 0).all(axis=1)] : 0이 되는 이상치, 결측치 모두 제거

- df.isnull().sum() : 결측치 제거 확인

- df.isnull().all(axis=1).sum()

- df[['col1 name','col2 name']].min()) 
- df[['col1 name','col2 name']].max()) 

- (df['col name'] < 0).any() # 음수값이 있는지 확인 후 전처리 (예: 변수 '가격'인데 음수는 있을 수 없음)​

### 4-3. 결측치 대체 (Imputation)
```shell
# 최빈값 대체
mode_cols = ['col1', 'col2', ..., 'col10']
for col in mode_cols:
    df[col] = df[col].fillna(df[col].mode()[0]) # .mean() : 평균값

# 회귀 대체
knn_cols = ['col1', 'col2', ..., 'col10']
for col in knn_cols:
    knn = KNeighborsRegressor(n_neighbors=10)
    numerical = df.select_dtypes(np.number)   # 수치형 변수 선택
    non_na_num_cols = numerical.loc[:, numerical.isna().sum() == 0].columns
    X = numerical.loc[numerical[col].isna() == False, non_na_num_cols] # 관측치 존재 컬럼
    y = numerical.loc[numerical[col].isna() == False, col] # 결측치 발생 컬럼
    X_test = numerical.loc[numerical[col].isna() == True, non_na_num_cols]
    knn.fit(X, y) # knn학습
    y_pred = knn.predict(X_test) # 결측치 예측
    df.loc[numerical[col].isna() == True, col] = y_pred  # 예측값으로 채우기
```

### 4-4. 이상치 확인 (Boxplot)

```shell
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, y='target_feature')
# sns.boxplot(data=df[["col1", "col2", "col3", "col4"]], orient="v")
plt.title("title of boxplot")
plt.ylabel("Price")
plt.show()  

# 이상치 제거를 위한 분위수 확인하기
Q3 = df['target_feature'].quantile(q=0.75)  # 75% 분위수
Q1 = df['target_feature'].quantile(q=0.25)  # 25% 분위수
IQR = Q3-Q1
print(Q1, Q3, IQR)
```
### 4-5. 이상치 대체
```shell
# 마이너스 값을 이전 기간 평균값으로 대체
imputation = int(df.loc["2024-01-01":"2024-03-01"]["col"].mean())
df.loc[df["col"] <= 0, "col"] = imputation  

# 특정값으로 대체
df.loc[df['year'] == 2526, 'year'] = 2026
```

## 5. 시각화

### 5-1. 변수 분포 확인(히스토그램) 

- 히스토그램

    sns.histplot(data=df, x=target_feature)  # kde 옵션: 막대 + 밀도 형태의 선으로도 보여줌

    plt.show()  

- log scale

    sns.histplot(data=df, x=target_feature, kde=True, log_scale=(False, True))

    plt.ylim(bottom=1)  # y축 최소값 1로 설정

    plt.show()  
​

### 5-2. 범주형 변수별 수치 데이터 확인(막대그래프) 

- Barplot 

    category_feature = 'category' # 막대그래프의 막대로 범주를 확인할 변수

    target_feature = 'lowest_monthly_earnings' # 막대그래프의 높이로 값을 확인할 변수


    barplot = sns.barplot(data=df x=category_feature, y=target_feature, color='C0', errorbar=None)

    loc, labels = plt.xticks() # matplotlib에서 x축 레이블 위치, 방향 설정

    barplot.set_xticklabels(labels, rotation=90)

    plt.title('Lowest Monthly Earnings per each category')

    plt.show()

```shell
# 선형 그래프
sns.lineplot(data=df['col'], label='Price')  # 주식가격 흐름 확인
# 산점도
sns.scatterplot(data=df['col'], x="xlabel", y="ylabel", color='indigo', marker='o', s=80)
```
​
### 5-3. 데이터의 상관관계 확인(수치형) 

- Correlation

    corr = df.corr(numeric_only=True)  # 피어슨 상관계수를 계산하여 행렬형태로 나타냄, numeric_only는 숫자형 변수만 계산하도록 하는 옵션.

    mask = np.ones_like(corr, dtype=bool) # figure에서 생략될 부분을 지정하는 mask 행렬 생성

    mask = np.triu(mask)

- heatmap

    plt.figure(figsize=(13,10)) 

    sns.heatmap(data=corr, annot=True, fmt='.2f', mask=mask, linewidths=.5, cmap='RdYlBu_r') # 히트맵 형태로 상관행렬 시각화

    plt.title('Correlation Matrix')

    plt.show()

### 5-4. 확률밀도추정 kde plot
```shell
# bar차트보다 직관적으로 두 데이터의 분포 비교
# 스케일링 후 밀도 비교 (표준화 vs 로버스트)
sns.kdeplot(data=df["Robust"], label="Robust", color="red", shade=True)
sns.kdeplot(data=df["Stdzation"], label="Stdzation", color="indigo", shade=True)
```
​
## 6. 데이터 변환

### 6-1. log 변환

- 분포의 꼬리가 길어 한쪽으로 치우칠 때, 정규분포에 가깝게 하기 위해 log scale로 변환
  
- from sklearn.preprocessing import scale

- log 변환
```shell
# 변수 선택
target_feature = 'lowest_monthly_earnings' 
data[f'log_{target_feature}'] = scale(np.log(df[target_feature]+1))   # log에 0값이 들어가는 것을 피하기 위해+1
df[log col] = np.log1p(df["col"]) # np.log1p=log(x+1) : 0 및 음수 값에 대한 대응을 위해

data[f'log_{target_feature}'].describe() # 로그변환한 변수의 요약통계량 보기
```
- 변환 전후 요약통계량 비교

    df[[target_feature, f'log_{target_feature}']].describe()

- 그래프로 비교

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))   # 데이터 분포 변환 전후를 비교하기 위해 2개의 plot 지정

    sns.histplot(data=df, x=target_feature, kde=True, ax=ax[0])  # 히스토그램으로 변환 전 데이터의 분포 보기

    ax[0].set_title('Before log transformation')​

    sns.histplot(data=df, x=f'log_{target_feature}', kde=True, ax=ax[1])    # 히스토그램으로 변환 후 데이터의 분포 보기

    ax[1].set_title('After log transformation')​

    plt.show()

  - 제곱근 변환 : df['sqrt col'] = np.sqrt(df['col'])
  - Box-Cox 변환 :
    ```shell
    lambdas = [[-3.0, -2.0, -1.0], [1.0, 2.0, 3.0]]
    df['boxcox col'] = boxcox1p(df[col], lambdas[i][j])
    ```


### 6-2. 표준화 : 평균 0, 표준편차 1

- from sklearn.preprocessing import StandardScaler

- scikit-learn패키지를 활용한 **standardization**

    target_feature = 'lowest_monthly_earnings'  # 스케일링을 진행할 변수를 선택

    standard_scaler = StandardScaler()

    df[f'standardized_{target_feature}'] = standard_scaler.fit_transform(df[[target_feature]])  # 표준화한 데이터를 새로운 변수에 저장

- 비교를 위해 표준화 전후 feature를 선택

    feature_original = df[target_feature]

    feature_standardized = df[f'standardized_{target_feature}']

- 표준화 전후의 평균, 표준편차 비교

    print(f"평균(mean)비교: {feature_original.mean():.7} vs {feature_standardized.mean():.7}")

    print(f"표준편차(standard deviation)비교: {feature_original.std():.7} vs {feature_standardized.std():.7}")

- 데이터 단위 변환 전후 비교

    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(15,6))  # 2개의 plot 지정

    sns.histplot(data=df, x=target_feature, kde=True, ax=ax[0])

    ax[0].set_title('Before standardization')

    sns.histplot(data=df, x=f'standardized_{target_feature}', kde=True, ax=ax[1])

    ax[1].set_title('After standardization')

    plt.show()

   - 로버스트 스케일링 : 중앙값에 가까울수록 0, 멀어질수록 큰값으로 변환
  ```shell
  robust_scaler = RobustScaler()
  df[["col1", "col2"]] = robust_scaler.fit_transform(df[["col1", "col2"]])
  ```

​
### 6-3. 정규화 : 0 ~ 1 사이의 값

- from sklearn.preprocessing import MinMaxScaler

- scikit-learn패키지를 활용한 **normalization**

    target_feature = 'lowest_monthly_earnings' # 스케일링을 진행할 변수 선택

    normalized_scaler = MinMaxScaler()

    df[f'normalized_{target_feature}'] = normalized_scaler.fit_transform(df[[target_feature]])  # 표준화한 데이터를 새로운 변수에 저장

- 비교를 위해 정규화 전후 feature를 선택

    feature_original = df[target_feature]

    feature_normalized = df[f'normalized_{target_feature}']

- 변환 전후의 최대, 최소값 비교

    print(f"최소값(min) 비교: {feature_original.min():.7} vs {feature_normalized.min():.7}")

    print(f"최대값(max) 비교: {feature_original.max():.7} vs {feature_normalized.max():.7}")

- 데이터 단위 변환 전후 비교 

    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(15,6))  # 2개의 plot 지정

    sns.histplot(data=df, x=target_feature, kde=True, ax=ax[0])

    ax[0].set_title('Before normalization')

    sns.histplot(data=df, x=f'normalized_{target_feature}', kde=True, ax=ax[1])

    ax[1].set_title('After normalization')

    plt.show()

### 6-4. 구간화 : 수치형을 범주형 변수로 변환
```shell
bins = [0, 300000, 500000, 1000000]
labels = ["Low", "Medium", "High"]
df["Bin"] = pd.cut(df["price"], bins=bins, labels=labels)
```

### 6-5. 범주형 -> 수치형 변수 변환
```shell
# 원핫인코딩
encoder = OneHotEncoder()
# 2차원 형태의 배열로 입력해야 하기 때문에 대괄호([])가 2개
onehot = encoder.fit_transform(onehot_data[["col"]])
# 배열형태로 전환
onehot_array = onehot.toarray()
onehot_df_sample["Onehot"] = onehot_array[:5].tolist()

# 레이블 인코딩
encoder = LabelEncoder()
label = encoder.fit_transform(label_df[["col"]])
label_df["Label"] = label

# 빈도 인코딩
# scikit-learn에는 없음
# 대신 category_encoders 패키지의 CountEncoder를 사용
encoder = CountEncoder(cols=['col'])
frequency = encoder.fit_transform(frequency_df['col'])
frequency_df["Frequency"] = frequency

# 타겟 인코딩
# price 열의 평균 구하기
target = target_df.groupby(df['col'])['price'].mean()
# map을 이용해 변환
target_df['Target'] = target_df['col'].map(target)
```

### 6-6. 파생변수
```shell
# 종목(code)로 집계 후 다음날 종가(Close)를 Target 변수에 삽입
OHLCV_data["Target"] = OHLCV_data.groupby("code")["Close"].shift(-1)
#.shift(-1): 한 칸 위로 이동. shuft(1): 한 칸 아래로

# 통계기반
# 일별 종가의 평균, 중앙값
close_stats = df.groupby(["Date", "industry"])["Close"].agg(["mean", "median"])
close_stats.columns = ["CloseMean", "CloseMedian"]
df = pd.merge(df, close_stats, how="inner", on=["industry", "Date"])

# 시간 관련
# Date컬럼 나누기
df["DateYear"] = df["Date"].dt.year
df["DateMonth"] = df["Date"].dt.month
df["DateDay"] = df["Date"].dt.day
# "00월"형태를 "월"빼고 정수만 남기
df["MonthInt"] = df["month"].str.replace('월', '').astype(int)

# 변수간 관계 활용
# 부대시설 유무 
df["has2ndFloor"] = df["2ndFlrSF"].apply(lambda x: 1 if x > 0 else 0)
df["hasGarage"] = df["GarageArea"].apply(lambda x: 1 if x > 0 else 0)
# 리모델링 연도 계
df['RemodAfterBuilt'] = df['YearRemodAdd'] - df['YearBuilt']
df['RemodAfterBuilt'] = df['RemodAfterBuilt'].clip(lower=0)  # 하향임계값 lower bound 설정
# 연면적 계산
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

# 변수 자체를 변환
# log
y_train = np.log1p(y_train)
# min-max scaling
for col in df.select_dtypes(np.number).columns:
    df[col] = MinMaxScaler().fit_transform(df[[col]])
# label encoding
for col in df.select_dtypes("object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])
```






