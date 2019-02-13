import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import copy
warnings.filterwarnings('ignore')
from matplotlib import style
style.use("ggplot")

from scipy import sparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder 

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import chi2, SelectPercentile,f_classif
import lightgbm as lgb


import xgboost as xgb
data = pd.read_pickle("../Feature/_featureEnineering_v5.pickle")
train = data[data["month_rent"]!=-1]
test = data[data["month_rent"]==-1]
X = train.drop(["month_rent"],axis=1).values
y = train["month_rent"].values
X_test = test.drop(["month_rent"],axis=1).values
cols = test.drop(["month_rent"],axis=1).columns

print(X.shape,y.shape,X_test.shape)

from sklearn.cross_validation import StratifiedKFold
seed_ls = []
# 五折交叉训练，构造五个模型
skf=list(StratifiedKFold(y, n_folds=5, shuffle=True, random_state=1024))
baseloss = []
loss = 0
for i, (train_index, test_index) in enumerate(skf):
    print("Fold", i)
    model = xgb.XGBRegressor(learning_rate=0.05, n_estimators=2000,booster='gbtree',max_depth=10,
                             seed=2018,num_thread=-1,bagging_fraction=0.8,colsample_bytree=0.9,subsample=0.8,reg_lambda=0.20)
    #
    #------------------------------------#
    xgb_model = model.fit(X[train_index], y[train_index],
                          eval_metric= 'rmse',
                          eval_set=[(X[train_index], y[train_index]), 
                                    (X[test_index], y[test_index])],early_stopping_rounds=100)
    baseloss.append(xgb_model.best_score)
    loss += xgb_model.best_score
    test_pred= xgb_model.predict(X_test, ntree_limit=xgb_model.best_iteration)
    if i==0:
        predict = test_pred
    else:
        predict = np.vstack((predict,test_pred))
print('logloss:', baseloss, loss/5)


submission= pd.DataFrame(predict.mean(axis=0))
test = pd.read_csv("../Data/test.csv")
submission["id"] = test["id"].values
submission.columns=["price","id"]
submission[["id","price"]].to_csv("../Result/_xgboost_线下_v5.1.2.csv",index=False,encoding="utf-8",sep=",")