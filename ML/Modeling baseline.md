# 모델링 baseline
## 1. 기본 ML 모델
1. Linear Regression: 데이터를 가장 잘 대변하는 최적의 회귀선 찾기
```shell
# 회귀 가정 체크 -> 변수간 독립
corr = X_train.corr()
sns.heatmap(corr)
plt.title("Correlation plot")
plt.show()
```
```shell
lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train) # sklearn based의 회귀 모델을 적합
# 위에서 적합된 모델로 MSE와 R^2를 계산.
y_pred_sk = lin_reg.predict(X_valid)
J_mse_sk = np.sqrt(mean_squared_error(y_pred_sk, Y_valid))
R_square_sk = lin_reg.score(X_valid,Y_valid)
display(f"RMSE : {J_mse_sk}")
display(f"R square : {R_square_sk}")
# 동일한 과정을 statsmodel 패키지로 사용. 조금 더 디테일
model = sm.OLS(Y_train, X_train)
result = model.fit()
print(result.summary())
```
```shell
# QQ plot
f,ax = plt.subplots(1,2,figsize=(10,4))
_,(_,_,r)= sp.stats.probplot((Y_valid - y_pred_sk),fit=True,plot=ax[0])
ax[0].set_title('Check for Multivariate Normality: \nQ-Q Plot')
# 등분산성 확인
sns.scatterplot(y = (Y_valid - y_pred_sk), x= y_pred_sk, ax = ax[1],color='r')
ax[1].set_title('Check for Homoscedasticity: \nResidual Vs Predicted')
# VIF 확인. 상관계수들과 관련 있음.
VIF = 1/(1- R_square_sk)
display(f"VIF: {VIF}")
```

2. KNN (K-Nearest Neighbor): 가까운 이웃에 위치한 K개의 데이터를 보고, 데이터가 속할 그룹을 판단하는 과정
```shell
knn_db=[]
for i in tqdm(range(3,15,2)):
    M=[]
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train, Y_train)
    y_pred = knn.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(Y_valid, y_pred))   # MSE를 K값에 따라 저장하도록 설정합니다.
    M.append(i)
    M.append(rmse)
    knn_db.append(M)
# 최적의 k값을 찾기 위한 장치. MSE가 가장 작게 나오는 모델을 적합시킨 K값을 최적의 값으로 설정
min=knn_db[0];
for i in range(len(knn_db)):
    if knn_db[i][1]<min[1]:
        min=knn_db[i];
n=min[0];
# 최적의 K 값이 반환
display(n)

# K값에 따른 MSE 변화를 시각화. K값이 올라갈수록 MSE가 증가.
x = [x[0] for x in knn_db]
y = [x[1] for x in knn_db]
# 시각화
plt.plot(x, y, color='red')
plt.title('K selection using MSE')
plt.xlabel('K')
plt.ylabel('MSE')
plt.show()
```
```shell
# 위에서 찾은 최적의 K값을 이용해 모델을 학습시켜 결과를 관찰.
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, Y_train)
# ytrainpredict_kk = knn.predict(X_train)     # 시간이 꽤 오래 걸림
ytestpredict_kk = knn.predict(X_valid)
# 회귀 관련 metric을 통해 train/valid의 모델 적합 결과를 관찰
display(f'RMSE Test: {np.sqrt(metrics.mean_squared_error(Y_valid, ytestpredict_kk))}')
# 모델 적합 결과를 시각화. (잔차플롯)
plt.scatter(ytestpredict_kk, ytestpredict_kk-Y_valid, c='limegreen', marker='s', edgecolors='white', s=35, alpha=0.9, label="Valid data")
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=ytestpredict_kk.min()-1, xmax=ytestpredict_kk.max()+1, lw=1, color='black')
plt.xlim([ytestpredict_kk.min()-1, ytestpredict_kk.max()+1])
plt.title('Residual plot from KNN Regression')
plt.show()
```
3. Decision Tree: 데이터 내 규칙을 찾아 Tree 구조로 데이터를 분류/회귀 하는 과정
```shell
# DecisionTreeRegressor 이용해 의사결정나무를 만듬.
dt = DecisionTreeRegressor(
    max_depth=10, # 트리의 깊이를 규제.
    random_state=1214, # 트리의 랜덤시드를 설정.
    min_samples_split=100 # 해당하는 샘플이 100개 이상이면 split하도록.
    )
dt.fit(X_train, Y_train)
ytrainpredict_df = dt.predict(X_train)
ytestpredict_df = dt.predict(X_valid)
```
```shell
# 회귀 관련 metric을 통해 train/valid의 모델 적합 결과를 관찰.
display(f"RMSE Train: {np.sqrt(metrics.mean_squared_error(Y_train, ytrainpredict_df))}")
display(f"RMSE Test: {np.sqrt(metrics.mean_squared_error(Y_valid, ytestpredict_df))}")
```
```shell
# 트리를 시각화. 트리의 깊이가 깊어지면 시각화에 시간이 오래걸릴 수 있음.
export_graphviz(dt, out_file ='tree.dot')
with open("tree.dot") as f:
    dot_graph = f.read()

pydot_graph = pydotplus.graph_from_dot_file("tree.dot")
Image(pydot_graph.create_png())

# 모델 적합 결과를 시각화. (잔차플롯)
plt.scatter(ytestpredict_df, ytestpredict_df-Y_valid, c='limegreen', marker='s', edgecolors='white', s=35, alpha=0.9, label="Valid data")
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=ytestpredict_df.min()-1, xmax=ytestpredict_df.max()+1, lw=1, color='black')
plt.xlim([ytestpredict_df.min()-1, ytestpredict_df.max()+1])
plt.title('Residual plot from DecisionTreeRegressor')
plt.show()

```
4. Random Forest: 
-배깅(Bagging)의 일종으로, 학습시키는 데이터 뿐 아니라 특성변수(X)들도 무작위로 선택해 트리를 생성하는 방법. 모델 학습 과정에서 서로 다른 N개의 Tree 생성. N개의 Tree에서 발생한 Output을 Voting(범주형, 분류문제)하거나, Average(연속형, 회귀문제)해 최종 Output 생성
→ High variance, Low bias 상황에서 분산(Variance) 감소에 도움
```shell
# RandomForestRegressor를 이용해 회귀 모델을 적합.
forest_rf = RandomForestRegressor(n_estimators=10, criterion='squared_error', random_state=1, n_jobs=-1)
forest_rf.fit(X_train, Y_train)
ytestpredict_rf = forest_rf.predict(X_valid)
```
```shell
# 회귀 관련 metric을 통해 train/valid의 모델 적합 결과를 관찰.
print(f'RMSE test: {np.sqrt(metrics.mean_squared_error(Y_valid, ytestpredict_rf))}')
# 모델 적합 결과를 시각화. (잔차플롯)
plt.scatter(ytestpredict_rf, ytestpredict_rf-Y_valid, c='limegreen', marker='s', edgecolors='white', s=35, alpha=0.9, label="Valid data")
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=ytestpredict_rf.min()-1, xmax=ytestpredict_rf.max()+1, lw=1, color='black')
plt.xlim([ytestpredict_rf.min()-1, ytestpredict_rf.max()+1])
plt.title('Residual plot from RandomForestRegressor')
plt.show()
```

## 2. 심화 ML 모델

1. LightGBM
-scikit-learn API 활용
```shell
# Regression모델을 지정.
gbm = lgb.LGBMRegressor(n_estimators=100) # 총 반복 횟수를 지정.

# 학습을 진행.
# scikit-learn api를 사용할 시, 별도의 lightgbm Dataset으로 변환을 거치지 않고 원데이터 형태로 사용.
%%time
gbm.fit(
    X_train, Y_train, # 학습 데이터를 입력.
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], # 평가셋을 지정.
    eval_metric ='rmse', # 평가과정에서 사용할 평가함수를 지정.
    callbacks=[lgb.log_evaluation(period=10, show_stdv=True)] .
    )

# 학습한 모델을 저장.
joblib.dump(gbm, 'lightgbm_sklearn_api.pkl')
# 저장한 모델을 불러옵니다.
gbm_trained = joblib.load('lightgbm_sklearn_api.pkl')
# 불러온 모델을 통해 추론을 진행.
predicts = gbm_trained.predict(X_valid)

display(f"추론 결과 샘플 : {predicts[:4]}")
# 학습에서 평가셋으로 사용한 세트를 이용한 추론성능 평가를 진행.
%%time
RMSE = mean_squared_error(Y_valid, predicts)**0.5

display(f"추론 결과 rmse : {RMSE}")
```
2. Parameters
- 2.1. boosting_type:
default는 gbdt로, gbdt, rf(random forest), dart(참고자료) 중 선택 가능.
rf(random forest)는 bagging방식을 기반으로 동작.
따라서 별도의 셋팅없이 rf를 사용할 경우, bagging_fraction 또는 feature_fraction 파라미터를 0초과 1미만의 값을 주어야 한다는 에러를 발생.
- 2.2. data_sample_strategy:
data sampling 방식을 지정하는 parameter로, default는 bagging입니다. bagging과 goss중 선택가능.
default값이 bagging이지만, "bagging_freq > 0"과 "bagging_fraction < 1.0"을 만족해야 적용.
결국 별도의 parameter 조합이 구성되지 않는 이상은 default로 data sampling은 적용되지 않도록 하기 위해, goss가 default값이 아니게 됨.
goss를 사용할 경우 topset과 randset의 비율은 top_rate, other_rate 라는 별도의 parameter를 통해 통제할 수 있으며, 각 default값은 0.2, 0.1.
- 2.3. max_depth:
트리의 최대 깊이를 제어하는 parameter로, default값은 20.
트리의 깊이가 깊어 질수록 한번의 학습으로 조절되어지는 트리가 많아지며, 과적합이 발생할 수 있음.
따라서, 과적합이 발생했다면 max_depth를 낮추면서 조절할 수 있음.
- 2.4. num_leaves:
트리의 leaf 수를 조절하는 parameter로 default값은 31.
2^(max_depth)보다 낮은 수를 사용가능하며, max_depth와 유사하게 복잡성을 제어하여 과적합을 줄일 수 있음.
- 2.5. min_data_in_leaf:
트리의 leaf가 가질 수 있는 최소 인스턴스 수를 조절하며 default값은 20.
큰 값을 주어 너무 깊은 tree가 구성되는 것을 피하면서 과적합을 방지할 수 있음.
너무 큰 값을 줄 경우, 반대로 underfitting이 발생할 수 있음.
- 2.6. feature_fraction:
default값은 1.0으로, 매 iteration에서 feature들에 대한 하위집합을 만들 비율을 정하는 parameter.
- 2.7. bagging_fraction:
default값은 1.0으로, feature_fraction과 유사하지만 feature가 아닌 data(row)의 하위집합을 무작위로 선택한다는 차이가 있음.
- 2.8. bagging_freq:
bagging_fraction으로 sampling의 비율을 정했지만, bagging_freq라는 parameter도 조절이 필요.
bagging_freq는 몇번의 반복마다 bagging을 다시 진행할지 조절할 수있으며, default는 0의 값을 가짐.
따라서, bagging_freq를 1이상의 값을 주지않는다면 bagging_fraction은 적용되지 않음.
```shell
# 과적합 방지 위한 튜닝
# num_leaves, min_data_in_leaf 모두 기존보다 높은 값, max_depth는 기존보다 낮은 값.
# early stopping을 적용하여 학습의 종료 조건, n_estimators는 가능한 큰 수.
%%time
gbm = lgb.LGBMRegressor(n_estimators=100000,                # early stopping을 적용하기에 적당히 많은 반복 횟수를 지정.
                        metric="rmse",
                        data_sample_strategy='goss',        
                        max_depth=12,                       # default값인 20에서 12로 변경.
                        num_leaves=62,                      # default값인 31에서 62으로 변경.
                        min_data_in_leaf=40                 # default값인 20에서 40으로 변경.
                        )
gbm.fit(X_train, Y_train,
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        eval_metric ='rmse',
        categorical_feature="auto",
        callbacks=[lgb.early_stopping(stopping_rounds=50),         # 50번동안 metirc의 개선이 없다면 학습을 중단.
                   lgb.log_evaluation(period=10, show_stdv=True)]  # 10번의 반복마다 평가점수를 로그에 나타냄.
)
```
```shell
# 학습한 모델을 저장.

joblib.dump(gbm, 'tuning_lightgbm_sklearn_api.pkl')
# 저장한 모델을 불러오기.

gbm_trained = joblib.load('tuning_lightgbm_sklearn_api.pkl')
# 불러온 모델을 통해 추론을 진행.

predicts = gbm_trained.predict(X_valid)
print(f"추론 결과 샘플 : {predicts[:5]}")

# 학습에서 평가셋으로 사용한 세트를 이용한 추론성능 평가를 진행.
%%time
RMSE = mean_squared_error(Y_valid, predicts)**0.5
print(f"추론 결과 rmse : {RMSE}")

# 그외) learning rate를 줄이면서 변화폭을 줄이는 방향으로 더욱 세밀한 최적값에 위치하도록 조절가능
```

3. 시각화
```shell
# 저장한 모델을 불러오기.
gbm_trained = joblib.load('tuning_lightgbm_sklearn_api.pkl')
# 학습된 gbm의 parameter들을 불러오기.
gbm_params = gbm_trained.get_params()
print(json.dumps(gbm_params, indent=4)) # 보기 편하게 나타내기 위해, json 형태로 변환.
# 학습결과를 시각화.
fig = plt.figure(figsize=(8, 8))
plt.subplot(1, 1, 1)
plt.plot(np.arange(len(list(gbm_trained.evals_result_['training']['rmse']))), list(gbm_trained.evals_result_['training']['rmse']), 'b-',
         label='Train Set')
plt.plot(np.arange(len(list(gbm_trained.evals_result_['training']['rmse']))), list(gbm_trained.evals_result_['valid_1']['rmse']), 'r-',
         label='Valid Set')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Rmse')
fig.tight_layout()
plt.show()
```
```shell
# 종목별 RMSE 시각화
code_li = list(X_valid['LEncodedCode'].unique())  # 종목리스트를 만듭니다.

valid_rmse_li = []                                # 종목별 rmse를 저장할 리스트를 선언.
for code in code_li:                              # 종목별로 rmse측정을 진행.
  valid_sets = X_valid.loc[X_valid['LEncodedCode']==code] # 해당 종목의 input을 가져오기기.

  valid_predicts = gbm_trained.predict(valid_sets, verbosity=-1) # 추론을 진행합니다.

  valid_rmse_li.append(mean_squared_error(Y_valid[valid_sets.index], valid_predicts)**0.5) # 추론결과로 rmse를 계산하고 저장.

# 종목별 rmse결과를 dataframe형태로 바꾸고, 오름차순으로 정렬.
valid_rmse_df = pd.DataFrame({'code':code_li, 'rmse':valid_rmse_li})
valid_rmse_df = valid_rmse_df.sort_values(by='rmse', ascending=True).reset_index(drop=True)

# 시각화.
plt.figure(figsize=(15,10))
barplot = sns.barplot(data=valid_rmse_df,         # 종목별 rmse를 barplot으로 시각화.
                      x="code",
                      y="rmse",
                      order=valid_rmse_df['code'])
    barplot.set(xticklabels=[])                       # 종목명은 시각화에서 제외.
barplot.set(xlabel=None)
sns.lineplot(x=valid_rmse_df['code'].index, y=[RMSE] * len(valid_rmse_df), label="Total_Valid_RMSE")  # 전체 validset에 대한 rmse를 lineplot으로 시각화.
plt.legend(loc='upper center')
plt.grid(True, axis='y')
plt.show()
```
```shell
# 종목별 예측 시각화
# 높게 나타난 3개의 종목을 추출.
high_rmse_sample = valid_rmse_df.tail(3) # 이전의 rmse기준으로 오름차순으로 정렬된 df에서 끝부분 3개.
# 각 종목별로 예측결과와 실제결과를 함께 시각화.
for idx, row in high_rmse_sample.iterrows():
  code_input = X_valid.index[X_valid['LEncodedCode'] == row['code']].tolist()  # 검증세트의 입력에서 해당 종목의 인덱스를 가져오기.

  plt.figure(figsize=(15,6))
  sns.lineplot(x=range(0, len(code_input)), y=np.asarray(Y_valid)[code_input], label="Actual", alpha=1.0)  #  실제 결과들 중 앞선 인덱스를 통해 해당 종목의 실제값만 가져오기.
  sns.lineplot(x=range(0, len(code_input)), y=predicts[code_input], label="Predict", alpha=1.0, linestyle='--') # 예측 결과들 중 앞선 인덱스를 통해 해당 종목의 예측값만 가져오기기.
  origin_code = code_label_encoder.inverse_transform([int(row['code'])])[0] # 종목을 label encoding하기 전의 원본코드값으로 변환.
  code_name = company_data.loc[company_data['code']==origin_code]['company'].values[0] # company data에서 원본코드값에 매칭되는 종목명을 가져오기.
  plt.title(f"Code : {code_name}, RMSE : {row['rmse']:.2f} - Prediction Close Price")
  plt.ylabel('Close')
  plt.legend()
  plt.grid(True)
  plt.show()
```
```shell
# 산업군별 RMSE 시각화
industry_li = list(X_valid['LEncodedIndustry'].unique())  # 산업군리스트를 만들기.

valid_rmse_li = []                                # 산업군별 rmse를 저장할 리스트를 선언.
for industry in industry_li:                              # 산업군별로 rmse측정을 진행.
  valid_sets = X_valid.loc[X_valid['LEncodedIndustry']==industry] # 해당 산업군의 input을 가져오기.

  valid_predicts = gbm_trained.predict(valid_sets, verbosity=-1) # 추론을 진행.

  valid_rmse_li.append(mean_squared_error(Y_valid[valid_sets.index], valid_predicts)**0.5) # 추론결과로 rmse를 계산하고 저장.
# 산업군별 rmse결과를 df형태로 바꾸고, 오름차순으로 정렬.
valid_rmse_df = pd.DataFrame({'industry':industry_li, 'rmse':valid_rmse_li})
valid_rmse_df = valid_rmse_df.sort_values(by='rmse', ascending=True).reset_index(drop=True)
# 시각화
plt.figure(figsize=(15,10))
barplot = sns.barplot(data=valid_rmse_df,         # 산업군별 rmse를 barplot으로 시각화.
                      x="industry",
                      y="rmse",
                      order=valid_rmse_df['industry'])
barplot.set(xticklabels=[])                       # 산업군명은 시각화에서 제외.
barplot.set(xlabel=None)
sns.lineplot(x=valid_rmse_df['industry'].index, y=[RMSE] * len(valid_rmse_df), label="Total_Valid_RMSE")  # 전체 validset에 대한 rmse를 lineplot으로 시각화.
plt.legend(loc='upper center')
plt.grid(True, axis='y')
plt.show()
```
```shell
# 산업군별 예측 시각화
high_rmse_sample = valid_rmse_df.tail(3)
for idx, row in high_rmse_sample.iterrows():
  Industry_input =   X_valid.loc[X_valid['LEncodedIndustry'] == row['industry']].sort_values(by=['LEncodedCode'], ascending=True).index.tolist()

  plt.figure(figsize=(15,5))
  sns.lineplot(x=range(0, len(Industry_input)), y=np.asarray(Y_valid)[Industry_input], label="Actual", alpha=1.0)  #  실제 결과들 중 앞선 인덱스를 통해 해당 산업군의 실제값만 가져오기.
  sns.lineplot(x=range(0, len(Industry_input)), y=predicts[Industry_input], label="Predict", alpha=1.0, linestyle='--') # 예측 결과들 중 앞선 인덱스를 통해 해당 산업군의 예측값만 가져오기.

  origin_industry = industry_label_encoder.inverse_transform([int(row['industry'])])[0] # 산업군을 label encoding하기 전의 원본산업군으로 변환.
  plt.title(f"Industry : {origin_industry}, RMSE : {row['rmse']:.2f} - Prediction Close Price")
  plt.ylabel('Close')
  plt.legend()
  plt.grid(True)
  plt.show() 
```
4. Feature importance 시각화
```shell
feat_imp = gbm_trained.feature_importances_
print(f"GBM Feature Importance \n\n {feat_imp}")

# 불러온 feature importance를 feature의 이름과 함께 dataframe 형태로 저장.
# 저장 후, 가장 영향력이 강한 feature순으로 정렬.
sorted_feat_imp = pd.Series(feat_imp, input_cols).sort_values(ascending=False)
print(sorted_feat_imp)
# 시각화
plt.figure(figsize=(16,6))
sorted_feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
```
5. Plot Tree: 트리의 각 노드가 어떻게 연결되어있는지 확인 -> Feature Engineering/Parameter튜닝 등 다음 개선과정을 탐색
```shell
lgb.plot_tree(gbm_trained,
              figsize=(50,20))
```









