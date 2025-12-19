# EDA baseline

## 1. DF 만들기

- df = pd.read_csv('{PATH / FILE NAME}', encoding='{utf-8, cp949}')


## 2. DF 확인

- df.columns : 컬럼 전체를 리스트로 반환

- df.index : 행 인덱스 정보 제공

- df.shape : 행, 열의 크기 

- df.info() : 컬럼 정보와 데이터 타입, 총 행의 갯수 요약하여 보기 

- df.describe() : 컬럼별 기초 통계량 확인

- df['col'].unique() : 중복 제거하고 리스트로 반환

- df['col'].value_counts() : 중복 제거하고 갯수 반환

- df['col'][:] : 지정한 컬럼의 데이터에 대하여 행을 슬라이싱하여 데이터 반환  

- df.iloc[row no, col no] : 지정한 행번호와 열번호(슬라이싱 가능)의 데이터를 반환

- df.loc[row no, ['col1 name', 'col2 name', ]] : 지정한 컬럼들을 기준으로 행번호(슬라이싱 가능)에 해당하는 데이터를 반환

- df.columns[:] : 슬라이싱 한 구간만큼의 컬럼명들을 리스트로 반환


## 3. 전처리 

### 3-1. 결측치 확인

- df.isnull(), df.isna()

- df.isnull().sum(axis=1) > 0 # axis = 1(행) null이 하나라도 있는 행을 불리안 인덱싱 반환
- df.isna().sum(axis=1) > 0 # axis = 1(행) 


### 3-2. 결측치 제거

- df_copy = df : 원본 데이터 보관 후 작업

- indices_to_keep = df.isnull().sum(axis=1) < 10 : 결측치 제거 대상을 제외하는 선정 

- df = df[indices_to_keep].copy() : 결측치를 제거하고 데이터 복사

- df['col name'].fillna(0, inplace=True) : na를 0으로 채워줌

- df.dropna(axis=0, inplace=True) : axis=0일 경우 결측치가 있는 행을, axis=1이면 열을 전부 제거

- df.isnull().sum() : 결측치 제거 확인

- df.isnull().all(axis=1).sum()

- df[['col1 name','col2 name']].min()) 
- df[['col1 name','col2 name']].max()) 

- (df['col name'] < 0).any() # 음수값이 있는지 확인 후 전처리 (예: 변수 '가격'인데 음수는 있을 수 없음)​


### 3-3. 이상치 확인 (Boxplot)

- import matplotlib.pyplot as plt

- import seaborn as sns

- sns.boxplot(data=df, y='target_feature')  

  plt.show()  

- 이상치 제거를 위한 분위수 확인하기

    Q3 = df['target_feature'].quantile(q=0.75)  # 75% 분위수

    Q1 = df['target_feature'].quantile(q=0.25)  # 25% 분위수

    IQR = Q3-Q1

    print(Q1, Q3, IQR)
​

## 4. 시각화

### 4-1. 변수 분포 확인(히스토그램) 

- 히스토그램

    sns.histplot(data=df, x=target_feature)  # kde 옵션: 막대 + 밀도 형태의 선으로도 보여줌

    plt.show()  

- log scale

    sns.histplot(data=df, x=target_feature, kde=True, log_scale=(False, True))

    plt.ylim(bottom=1)  # y축 최소값 1로 설정

    plt.show()  
​

### 4-2. 범주형 변수별 수치 데이터 확인(막대그래프) 

- Barplot 

    category_feature = 'category' # 막대그래프의 막대로 범주를 확인할 변수

    target_feature = 'lowest_monthly_earnings' # 막대그래프의 높이로 값을 확인할 변수


    barplot = sns.barplot(data=df x=category_feature, y=target_feature, color='C0', errorbar=None)

    loc, labels = plt.xticks() # matplotlib에서 x축 레이블 위치, 방향 설정

    barplot.set_xticklabels(labels, rotation=90)

    plt.title('Lowest Monthly Earnings per each category')

    plt.show()

​
### 4-3 데이터의 상관관계 확인(수치형) 

- Correlation

    corr = df.corr(numeric_only=True)  # 피어슨 상관계수를 계산하여 행렬형태로 나타냄, numeric_only는 숫자형 변수만 계산하도록 하는 옵션.

    mask = np.ones_like(corr, dtype=bool) # figure에서 생략될 부분을 지정하는 mask 행렬 생성

    mask = np.triu(mask)

- heatmap

    plt.figure(figsize=(13,10)) 

    sns.heatmap(data=corr, annot=True, fmt='.2f', mask=mask, linewidths=.5, cmap='RdYlBu_r') # 히트맵 형태로 상관행렬 시각화

    plt.title('Correlation Matrix')

    plt.show()

​
## 5. 데이터 변환

### 5-1. log 변환

- from sklearn.preprocessing import scale

- log 변환

    target_feature = 'lowest_monthly_earnings' # 변수 선택

    data[f'log_{target_feature}'] = scale(np.log(df[target_feature]+1))   # log에 0값이 들어가는 것을 피하기 위해+1

    data[f'log_{target_feature}'].describe() # 로그변환한 변수의 요약통계량 보기

- 변환 전후 요약통계량 비교

    df[[target_feature, f'log_{target_feature}']].describe()

- 그래프로 비교

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))   # 데이터 분포 변환 전후를 비교하기 위해 2개의 plot 지정

    sns.histplot(data=df, x=target_feature, kde=True, ax=ax[0])  # 히스토그램으로 변환 전 데이터의 분포 보기

    ax[0].set_title('Before log transformation')​

    sns.histplot(data=df, x=f'log_{target_feature}', kde=True, ax=ax[1])    # 히스토그램으로 변환 후 데이터의 분포 보기

    ax[1].set_title('After log transformation')​

    plt.show()


### 5-2. 표준화 : 평균 0, 분산 1

- from sklearn.preprocessing import StandardScaler

- scikit-learn패키지를 활용한 **standardization**

    target_feature = 'lowest_monthly_earnings'# 스케일링을 진행할 변수를 선택

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

​
### 5-3. 정규화 : 0 ~ 1 사이의 값

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
