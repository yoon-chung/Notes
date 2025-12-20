# Linear Regression baseline

```shell
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```
```shell
# 당뇨병 데이터셋을 로드
diabetes = load_diabetes(scaled=False)

# 데이터셋의 "설명"(description)섹션 확인
print(diabetes["DESCR"])
```
```shell
data = diabetes["data"]
data = pd.DataFrame(data, columns=diabetes["feature_names"])

# x : feature_names : feature별 평균, 표준편차 계산
fts_mean = data.mean(axis=0)
fts_std = data.std(axis=0)

# Standardization : 평균 0, 표준편차 1
data = (data - fts_mean) / fts_std

# y : target : 평균, 표준편차 계산
tgt_mean = target.mean()
tgt_std = target.std()

# Standardization : 평균 0, 표준편차 1
target = (target - tgt_mean) / tgt_std
```
```shell
random_state = 1234 # 재현성

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=random_state)

print(f"train data: {train_data.shape}")
print(f"train target: {train_target.shape}")
print(f"test data: {test_data.shape}")
print(f"test target: {test_target.shape}")
```
```shell
multi_regressor = LinearRegression()  # 초기화
multi_regressor.fit(train_data, train_target)  # 학습

print("intercept: ", multi_regressor.intercept_)
print("coefficients: ", multi_regressor.coef_)
```
```shell
multi_train_pred = multi_regressor.predict(train_data)
multi_test_pred = multi_regressor.predict(test_data)

multi_train_mse = mean_squared_error(multi_train_pred, train_target)
multi_test_mse = mean_squared_error(multi_test_pred, test_target)
print(f"Train MSE: {multi_train_mse:.5f}")
print(f"Test MSE: {multi_test_mse:.5f}")
```
```shell
plt.figure(figsize=(4, 4))

plt.xlabel("target")
plt.ylabel("prediction")

y_pred = multi_regressor.predict(test_data)
plt.plot(test_target, y_pred, '.')

x = np.linspace(-2.5, 2.5, 10)
y = x
plt.plot(x, y)

plt.show()
```
```shell
corr = data.corr(numeric_only=True)

 #매트릭스 우측 상단을 모두 True 인 1로, 하단을 False 인 0으로 변환, True/False mask 배열로 변환
mask = np.ones_like(corr, dtype=bool)  
mask = np.triu(mask) 

plt.figure(figsize=(13,10))
sns.heatmap(data=corr, annot=True, fmt='.2f', mask=mask, linewidths=.5, cmap='RdYlBu_r')
plt.title('Correlation Matrix')
plt.show()
```
```shell
x_feature = "s3"
y_feature = "s2"

simple_regressor = LinearRegression()  # 초기화
simple_regressor.fit(data[[x_feature]], data[[y_feature]])  # 학습

coef = simple_regressor.coef_  # 회귀계수
print(coef)
```
```shell
plt.figure(figsize=(4, 4))
plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.plot(data[[x_feature]], data[[y_feature]], ".")

x_min, x_max = -5, 5
plt.xlim(x_min, x_max)

x = np.linspace(x_min, x_max, 10)
y = coef.item() * x 
plt.plot(x, y)

plt.show()
```
