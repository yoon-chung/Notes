# 데이터셋 분할 baseline

## 1. Holdout
```shell
holdout_X_train, holdout_X_valid, holdout_Y_train, holdout_Y_valid = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True) # default shuffle : True
```
```shell
display(f"Train Input : {holdout_X_train.shape}")
display(f"Train Target : {holdout_Y_train.shape}")
display(f"Valid Input : {holdout_X_valid.shape}")
display(f"Valid Target : {holdout_Y_valid.shape}")
display(f"Test Input : {X_test.shape}")
display(f"Valid Target : {Y_test.shape}")

# LightGBM을 선언. 학습은 총 1000번을 반복 
# 데이터셋 분할에 따른 성능비교를 위해 별도의 Parameter 튜닝은 적용하지 않고 기본값을 사용.
gbm = lgb.LGBMRegressor(n_estimators=1000)

%%time
gbm.fit(holdout_X_train, holdout_Y_train,                                                
        eval_set=[(holdout_X_train, holdout_Y_train), (holdout_X_valid, holdout_Y_valid)],        eval_metric ='rmse',                                                                       callbacks=[lgb.early_stopping(stopping_rounds=10),                                                    lgb.log_evaluation(period=10, show_stdv=True)]                          # 매 iteration마다 학습결과를 출력
)
```
```shell
joblib.dump(gbm, 'holdout_gbm.pkl') # 모델 저장
gbm_trained = joblib.load('holdout_gbm.pkl') # 모델 불러오기
predicts = gbm_trained.predict(X_test) # 추론 진행
RMSE = mean_squared_error(Y_test, predicts)**0.5
print(f"Test rmse : {RMSE}")
```

## 2. K-Fold
```shell
kf = KFold(n_splits=5)
# 학습 데이터를 Kfold로 나눔.
train_folds = kf.split(X_train, Y_train)
display(train_folds)
```
```shell
# 학습을 진행.
%%time
fold_save_files = []

for fold_idx, (train_idx, valid_idx) in enumerate(train_folds):
    print(f"--------{fold_idx}번째 fold의 학습을 시작합니다.--------")

    # index를 통해 fold의 학습세트를 가져옵니다.
    X_train_fold = X_train.iloc[train_idx, :]
    Y_train_fold = Y_train[train_idx]

    # index를 통해 fold의 평가세트를 가져옵니다.
    X_valid_fold = X_train.iloc[valid_idx, :]
    Y_valid_fold = Y_train[valid_idx]

    # fold의 데이터로 학습을 진행.
    gbm = lgb.LGBMRegressor(n_estimators=1000)
    gbm.fit(X_train_fold, Y_train_fold,                                            
        eval_set=[(X_train_fold, Y_train_fold), (X_valid_fold, Y_valid_fold)], 
        eval_metric ='rmse',                                                                      callbacks=[lgb.early_stopping(stopping_rounds=10),                                                    lgb.log_evaluation(period=10, show_stdv=True)]                           
    )

    # 각 fold별 학습한 모델을 저장.
    file_name = f"kfold{fold_idx}_gbm.pkl"
    joblib.dump(gbm, file_name)
    print(f"--------{fold_idx}번째 fold는 {file_name}에 저장되었습니다.--------\n\n")
    fold_save_files.append(file_name)
```
```shell
# 저장한 학습모델들을 불러와, Testset에 대한 추론을 진행.
# 각 fold의 예측결과를 평균을 취하는 방식으로 진행.
total_predicts = np.zeros(len(X_test))

for file_name in fold_save_files:
    gbm_trained = joblib.load(file_name)
    fold_predicts = gbm_trained.predict(X_test)

    # 각 fold의 rmse를 측정.
    RMSE = mean_squared_error(Y_test, fold_predicts)**0.5
    display(f"{file_name} - Test rmse : {RMSE}")

    total_predicts += fold_predicts / len(fold_save_files)

RMSE = mean_squared_error(Y_test, total_predicts)**0.5
display(f"최종 Test rmse : {RMSE}")

# 위 학습 로그에서 검증세트를 기준으로 rmse가 가장 낮은 3개의 모델을 선택후 추론을 진행.
top_3_files = ["kfold1_gbm.pkl", "kfold3_gbm.pkl", "kfold4_gbm.pkl"]
total_predicts = np.zeros(len(X_test))

for file_name in top_3_files:
    gbm_trained = joblib.load(file_name)
    fold_predicts = gbm_trained.predict(X_test)

    # 각 fold의 rmse를 측정.
    RMSE = mean_squared_error(Y_test, fold_predicts)**0.5
    display(f"{file_name} - Test rmse : {RMSE}")

    total_predicts += fold_predicts / len(top_3_files)

RMSE = mean_squared_error(Y_test, total_predicts)**0.5
display(f"최종 Test rmse : {RMSE}")
```


## 3. Stratified K-Fold
```shell
kf = StratifiedKFold(n_splits=5)
# Target값을 기준으로 분포에 따라 1000등분.
# Stratified KFold에서 각 구간을 기준으로 Y의 비율을 동일하게 가져감.
# 0, ... 9 = 0 / 10, ..., 19 = 1 / ..
cut_Y_train = pd.cut(Y_train,
                     1000, # 데이터를 최소 최대 구간으로 1000등분
                     labels=False)
# 학습 데이터를 Stratified Kfold로 나눔.
train_folds = kf.split(X_train, cut_Y_train)
display(train_folds)
```
```shell
# 학습을 진행.
%%time
fold_save_files = []

for fold_idx, (train_idx, valid_idx) in enumerate(train_folds):
    print(f"--------{fold_idx}번째 fold의 학습을 시작합니다.--------")

    # index를 통해 fold의 학습세트를 가져옴.
    X_train_fold = X_train.iloc[train_idx, :]
    Y_train_fold = Y_train[train_idx]

    # index를 통해 fold의 평가세트를 가져옴.
    X_valid_fold = X_train.iloc[valid_idx, :]
    Y_valid_fold = Y_train[valid_idx]

    # fold의 데이터로 학습을 진행.
    gbm = lgb.LGBMRegressor(n_estimators=1000)
    gbm.fit(X_train_fold, Y_train_fold, # 학습 데이터를 입력.
        eval_set=[(X_train_fold, Y_train_fold), (X_valid_fold, Y_valid_fold)], # 평가셋을 지정.
        eval_metric ='rmse', # 평가과정에서 사용할 평가함수를 지정.
        callbacks=[lgb.early_stopping(stopping_rounds=10) # 10번의 성능향상이 없을 경우, 학습을 멈춤
                   lgb.log_evaluation(period=10, show_stdv=True)] # 매 iteration마다 학습결과를 출력.
    )

    # 각 fold별 학습한 모델을 저장.
    file_name = f"Stratified_kfold{fold_idx}_gbm.pkl"
    joblib.dump(gbm, file_name)
    display(f"--------{fold_idx}번째 fold는 {file_name}에 저장되었습니다.--------\n\n")
    fold_save_files.append(file_name)
```
```shell
# 저장한 학습모델들을 불러와, Testset에 대한 추론을 진행.
# 각 fold의 예측결과를 평균을 취하는 방식으로 진행.
total_predicts = np.zeros(len(X_test))

for file_name in fold_save_files:
    gbm_trained = joblib.load(file_name)
    fold_predicts = gbm_trained.predict(X_test)

    # 각 fold의 rmse를 측정.
    RMSE = mean_squared_error(Y_test, fold_predicts)**0.5
    display(f"{file_name} - Test rmse : {RMSE}")

    total_predicts += fold_predicts / len(fold_save_files)

RMSE = mean_squared_error(Y_test, total_predicts)**0.5
display(f"최종 Test rmse : {RMSE}")

# 위 학습 로그에서 검증세트를 기준으로 rmse가 가장 낮은 3개의 모델을 선택후 추론을 진행.
top_3_files = ["Stratified_kfold1_gbm.pkl", "Stratified_kfold2_gbm.pkl", "Stratified_kfold3_gbm.pkl"]
total_predicts = np.zeros(len(X_test))

for file_name in top_3_files:
    gbm_trained = joblib.load(file_name)
    fold_predicts = gbm_trained.predict(X_test)

    # 각 fold의 rmse를 측정.
    RMSE = mean_squared_error(Y_test, fold_predicts)**0.5
    display(f"{file_name} - Test rmse : {RMSE}")

    total_predicts += fold_predicts / len(top_3_files)

RMSE = mean_squared_error(Y_test, total_predicts)**0.5
display(f"최종 Test rmse : {RMSE}")
```

## 4. Group K-Fold
```shell
groups = list(X_train['LEncodedCode'].astype('int')) # 종목(code)를 Group을 나누는 기준으로 사용
kf = GroupKFold(n_splits=5)  # 함수 선언
# 학습 데이터를 GroupKfold로 나눔.
train_folds = kf.split(X_train, Y_train, groups=groups)
display(train_folds)
```
```shell
# 학습을 진행.
%%time
fold_save_files = []

for fold_idx, (train_idx, valid_idx) in enumerate(train_folds):
    display(f"--------{fold_idx}번째 fold의 학습을 시작합니다.--------")

    # index를 통해 fold의 학습세트를 가져옴.
    X_train_fold = X_train.iloc[train_idx, :]
    Y_train_fold = Y_train[train_idx]

    # index를 통해 fold의 평가세트를 가져옴.
    X_valid_fold = X_train.iloc[valid_idx, :]
    Y_valid_fold = Y_train[valid_idx]

    # fold의 데이터로 학습을 진행.
    gbm = lgb.LGBMRegressor(n_estimators=1000)
    gbm.fit(X_train_fold, Y_train_fold,                                               # 학습 데이터를 입력.
        eval_set=[(X_train_fold, Y_train_fold), (X_valid_fold, Y_valid_fold)], # 평가셋을 지정.
        eval_metric ='rmse',                                                                       callbacks=[lgb.early_stopping(stopping_rounds=10),                                                    lgb.log_evaluation(period=10, show_stdv=True)]                          
    )

    # 각 fold별 학습한 모델을 저장.
    file_name = f"groupd_kfold{fold_idx}_gbm.pkl"
    joblib.dump(gbm, file_name)
    display(f"--------{fold_idx}번째 fold는 {file_name}에 저장되었습니다.--------\n\n")
    fold_save_files.append(file_name)
```
```shell
# 저장한 학습모델들을 불러와, Testset에 대한 추론을 진행.
# 각 fold의 예측결과를 평균을 취하는 방식으로 진행.
total_predicts = np.zeros(len(X_test))

for file_name in fold_save_files:
    gbm_trained = joblib.load(file_name)
    fold_predicts = gbm_trained.predict(X_test)

    # 각 fold의 rmse를 측정.
    RMSE = mean_squared_error(Y_test, fold_predicts)**0.5
    display(f"{file_name} - Test rmse : {RMSE}")

    total_predicts += fold_predicts / len(fold_save_files)

RMSE = mean_squared_error(Y_test, total_predicts)**0.5
display(f"최종 Test rmse : {RMSE}")

# 위 학습 로그에서 검증세트를 기준으로 rmse가 가장 낮은 3개의 모델을 선택후 추론을 진행.
top_3_files = ["groupd_kfold0_gbm.pkl", "groupd_kfold2_gbm.pkl", "groupd_kfold4_gbm.pkl"]
total_predicts = np.zeros(len(X_test))

for file_name in top_3_files:
    gbm_trained = joblib.load(file_name)
    fold_predicts = gbm_trained.predict(X_test)

    # 각 fold의 rmse를 측정.
    RMSE = mean_squared_error(Y_test, fold_predicts)**0.5
    display(f"{file_name} - Test rmse : {RMSE}")

    total_predicts += fold_predicts / len(top_3_files)

RMSE = mean_squared_error(Y_test, total_predicts)**0.5
display(f"최종 Test rmse : {RMSE}")
```
```shell
groups = list(X_train['LEncodedIndustry'].astype('int'))  # 산업군(Industry)를 Group을 나누는 기준으로 사용.
kf = GroupKFold(n_splits=5)  # 함수 선언
# 학습 데이터를 GroupKfold로 나눔.
train_folds = kf.split(X_train, Y_train, groups=groups)
display(train_folds)
```
```shell
# 학습을 진행.
%%time
fold_save_files = []

for fold_idx, (train_idx, valid_idx) in enumerate(train_folds):
    display(f"--------{fold_idx}번째 fold의 학습을 시작합니다.--------")

    # index를 통해 fold의 학습세트를 가져옴.
    X_train_fold = X_train.iloc[train_idx, :]
    Y_train_fold = Y_train[train_idx]

    # index를 통해 fold의 평가세트를 가져옴.
    X_valid_fold = X_train.iloc[valid_idx, :]
    Y_valid_fold = Y_train[valid_idx]

    # fold의 데이터로 학습을 진행.
    gbm = lgb.LGBMRegressor(n_estimators=1000)
    gbm.fit(X_train_fold, Y_train_fold,                                               # 학습 데이터를 입력.
        eval_set=[(X_train_fold, Y_train_fold), (X_valid_fold, Y_valid_fold)], # 평가셋을 지정.
        eval_metric ='rmse',                                                        # 평가과정에서 사용할 평가함수를 지정.
        callbacks=[lgb.early_stopping(stopping_rounds=10),                                                     lgb.log_evaluation(period=10, show_stdv=True)]                          
    )

    # 각 fold별 학습한 모델을 저장.
    file_name = f"industry_groupd_kfold{fold_idx}_gbm.pkl"
    joblib.dump(gbm, file_name)
    display(f"--------{fold_idx}번째 fold는 {file_name}에 저장되었습니다.--------\n\n")
    fold_save_files.append(file_name)
```
```shell
# 저장한 학습모델들을 불러와, Testset에 대한 추론을 진행.
# 각 fold의 예측결과를 평균을 취하는 방식으로 진행.
total_predicts = np.zeros(len(X_test))

for file_name in fold_save_files:
    gbm_trained = joblib.load(file_name)
    fold_predicts = gbm_trained.predict(X_test)

    # 각 fold의 rmse를 측정.
    RMSE = mean_squared_error(Y_test, fold_predicts)**0.5
    display(f"{file_name} - Test rmse : {RMSE}")

    total_predicts += fold_predicts / len(fold_save_files)

RMSE = mean_squared_error(Y_test, total_predicts)**0.5
display(f"최종 Test rmse : {RMSE}")

# 위 학습 로그에서 검증세트를 기준으로 rmse가 가장 낮은 3개의 모델을 선택후 추론을 진행.
top_3_files = ["industry_groupd_kfold0_gbm.pkl", "industry_groupd_kfold1_gbm.pkl", "industry_groupd_kfold2_gbm.pkl"]
total_predicts = np.zeros(len(X_test))

for file_name in top_3_files:
    gbm_trained = joblib.load(file_name)
    fold_predicts = gbm_trained.predict(X_test)

    # 각 fold의 rmse를 측정.
    RMSE = mean_squared_error(Y_test, fold_predicts)**0.5
    display(f"{file_name} - Test rmse : {RMSE}")

    total_predicts += fold_predicts / len(top_3_files)

RMSE = mean_squared_error(Y_test, total_predicts)**0.5
display(f"최종 Test rmse : {RMSE}")
```

## 5. Time-Series Split
```shell
kf = TimeSeriesSplit(n_splits=5)   # 함수 선언
# 학습 데이터를 TimeSeriesSplit로 나눔
train_folds = kf.split(X_train, Y_train)
display(train_folds)
```
```shell
# 학습을 진행.
%%time
fold_save_files = []

for fold_idx, (train_idx, valid_idx) in enumerate(train_folds):
    display(f"--------{fold_idx}번째 fold의 학습을 시작합니다.--------")

    # index를 통해 fold의 학습세트를 가져옴.
    X_train_fold = X_train.iloc[train_idx, :]
    Y_train_fold = Y_train[train_idx]

    # index를 통해 fold의 평가세트를 가져옴.
    X_valid_fold = X_train.iloc[valid_idx, :]
    Y_valid_fold = Y_train[valid_idx]

    # fold의 데이터로 학습을 진행.
    gbm = lgb.LGBMRegressor(n_estimators=1000)
    gbm.fit(X_train_fold, Y_train_fold,                                      
        eval_set=[(X_train_fold, Y_train_fold), (X_valid_fold, Y_valid_fold)], 
        eval_metric ='rmse',                                                                     callbacks=[lgb.early_stopping(stopping_rounds=10),                                                  lgb.log_evaluation(period=10, show_stdv=True)]                          
    )

    # 각 fold별 학습한 모델을 저장.
    file_name = f"timeseries_fold{fold_idx}_gbm.pkl"
    joblib.dump(gbm, file_name)
    display(f"--------{fold_idx}번째 fold는 {file_name}에 저장되었습니다.--------\n\n")
    fold_save_files.append(file_name)
```
```shell
# 저장한 학습모델들을 불러와, Testset에 대한 추론을 진행.
# 각 fold의 예측결과를 평균을 취하는 방식으로 진행.
total_predicts = np.zeros(len(X_test))

for file_name in fold_save_files:
    gbm_trained = joblib.load(file_name)
    fold_predicts = gbm_trained.predict(X_test)

    # 각 fold의 rmse를 측정.
    RMSE = mean_squared_error(Y_test, fold_predicts)**0.5
    display(f"{file_name} - Test rmse : {RMSE}")

    total_predicts += fold_predicts / len(fold_save_files)

RMSE = mean_squared_error(Y_test, total_predicts)**0.5
display(f"최종 Test rmse : {RMSE}")

# 위 학습 로그에서 검증세트를 기준으로 rmse가 가장 낮은 3개의 모델을 선택후 추론을 진행.
top_3_files = ["timeseries_fold0_gbm.pkl", "timeseries_fold2_gbm.pkl", "timeseries_fold4_gbm.pkl"]
total_predicts = np.zeros(len(X_test))

for file_name in top_3_files:
    gbm_trained = joblib.load(file_name)
    fold_predicts = gbm_trained.predict(X_test)

    # 각 fold의 rmse를 측정.
    RMSE = mean_squared_error(Y_test, fold_predicts)**0.5
    display(f"{file_name} - Test rmse : {RMSE}")

    total_predicts += fold_predicts / len(top_3_files)

RMSE = mean_squared_error(Y_test, total_predicts)**0.5
display(f"최종 Test rmse : {RMSE}")
```

## 5.1. 시간정렬_Time-Series Split 
```shell
kf = TimeSeriesSplit(n_splits=5)
# 시간순 정렬을 위해 이전에 저장해두었던 시간정보를 붙임.
X_train['Date'] = date_list

X_train = X_train.sort_values(by='Date') # 시간순으로 정렬.
Y_train = Y_train.reindex(X_train.index) # 정렬된 X_train의 인덱스에 맞추어 Y_train도 정렬.

X_train = X_train.reset_index(drop=True) # 인덱스를 재정렬.
Y_train = Y_train.reset_index(drop=True)

del X_train['Date'] # 시간에 대한 정보를 지움.

# 학습 데이터를 TimeSeriesSplit로 나눔.
train_folds = kf.split(X_train, Y_train)
display(train_folds)
```
```shell
# 학습을 진행.
%%time
fold_save_files = []

for fold_idx, (train_idx, valid_idx) in enumerate(train_folds):
    display(f"--------{fold_idx}번째 fold의 학습을 시작합니다.--------")

    # index를 통해 fold의 학습세트를 가져옴.
    X_train_fold = X_train.iloc[train_idx, :]
    Y_train_fold = Y_train[train_idx]

    # index를 통해 fold의 평가세트를 가져옴.
    X_valid_fold = X_train.iloc[valid_idx, :]
    Y_valid_fold = Y_train[valid_idx]

    # fold의 데이터로 학습을 진행.
    gbm = lgb.LGBMRegressor(n_estimators=1000)
    gbm.fit(X_train_fold, Y_train_fold,                                              
        eval_set=[(X_train_fold, Y_train_fold), (X_valid_fold, Y_valid_fold)],
        eval_metric ='rmse',                                                                       callbacks=[lgb.early_stopping(stopping_rounds=10),                                                     lgb.log_evaluation(period=10, show_stdv=True)]                          
    )

    # 각 fold별 학습한 모델을 저장.
    file_name = f"timeseries_fold{fold_idx}_gbm.pkl"
    joblib.dump(gbm, file_name)
    display(f"--------{fold_idx}번째 fold는 {file_name}에 저장되었습니다.--------\n\n")
    fold_save_files.append(file_name)
```
```shell
# 저장한 학습모델들을 불러와, Testset에 대한 추론을 진행.
# 각 fold의 예측결과를 평균을 취하는 방식으로 진행.
total_predicts = np.zeros(len(X_test))

for file_name in fold_save_files:
    gbm_trained = joblib.load(file_name)
    fold_predicts = gbm_trained.predict(X_test)

    # 각 fold의 rmse를 측정.
    RMSE = mean_squared_error(Y_test, fold_predicts)**0.5
    display(f"{file_name} - Test rmse : {RMSE}")

    total_predicts += fold_predicts / len(fold_save_files)

RMSE = mean_squared_error(Y_test, total_predicts)**0.5
display(f"최종 Test rmse : {RMSE}")

# 위 학습 로그에서 검증세트를 기준으로 rmse가 가장 낮은 3개의 모델을 선택후 추론을 진행.
top_3_files = ["timeseries_fold2_gbm.pkl", "timeseries_fold3_gbm.pkl", "timeseries_fold4_gbm.pkl"]
total_predicts = np.zeros(len(X_test))

for file_name in top_3_files:
    gbm_trained = joblib.load(file_name)
    fold_predicts = gbm_trained.predict(X_test)

    # 각 fold의 rmse를 측정.
    RMSE = mean_squared_error(Y_test, fold_predicts)**0.5
    display(f"{file_name} - Test rmse : {RMSE}")

    total_predicts += fold_predicts / len(top_3_files)

RMSE = mean_squared_error(Y_test, total_predicts)**0.5
display(f"최종 Test rmse : {RMSE}")
```

