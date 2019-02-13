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

def readData():
    train = pd.read_csv("../Data/train.csv")
    test = pd.read_csv("../Data/test.csv")
    print(train.shape,test.shape)
    columns = ["time","department","rent_num","floor","total_floor","square","direction","living_state","num_bedroom",
          "num_living_room","num_bath_room","rent_type","district","position","metro_line","station","distance","decoration","month_rent"]
    #------------------------------
    train.columns = columns
    test.columns = ["id"] + columns[:-1]
    #-------------------------------
    train = train.drop("time",axis=1).reset_index()
    train = train.rename(columns = {"index":"id"})
    test["month_rent"] = -1
    data = pd.concat([train,test[train.columns]],axis=0)
    data = data.reset_index(drop=True)
    return data


def basicClean(df=None,is_del=True):
    data = copy.deepcopy(df)
    print("handle the null feature")
    if is_del:
        data = data.drop(["living_state","rent_type","decoration"],axis=1)
    else:
        for i in ["living_state","rent_type","decoration"]:
            data[i] = pd.factorize(data[i])[0]
    #-----异常值清洗square
    #data.loc[data["square"]>=0.1,"square"] = data.loc[data["square"]<0.1,"square"].mean()

    #--------空值处理-------------
    rent_median = data["rent_num"].median()
    print(rent_median)
    data["rent_num"] = data["rent_num"].fillna(data["rent_num"].median())
    data["distance"] = data["distance"].fillna(0)
    data["metro_line"] = data["metro_line"].fillna("none")
    data["station"] = data["station"].fillna("none")
    data["district"]  =data["district"].fillna("none")
    data["position"]  =data["position"].fillna("none")
    #-----log变换-----------
    log_cols = ["rent_num","square","distance","total_floor"]
    print(data.dtypes)
    for i in log_cols:
        data['log_'+i] = data[i].map(lambda x:np.log(x+0.0000001))
    #顺序变化
    enc_cols =["department","direction","position","station","metro_line","district"]
    for i in enc_cols:
        data[i] = pd.factorize(data[i])[0]
    return data  

def rankFeature(fea):
    data = readData()
    data = basicClean(data)
    data = data[[fea,"month_rent"]]
    train = data[data["month_rent"]!=-1]
    test = data[data["month_rent"]==-1]
    tmp = train.groupby(fea,as_index=False)[["month_rent"]].median()
    tmp["rank"] = tmp["month_rent"].rank()
    maper = pd.Series(tmp["rank"],index=tmp[fea])
    #train
    train[fea+"_rank"] = train[fea].map(maper)
    test[fea+"_rank"] = test[fea].map(maper)
    data = pd.concat([train,test],axis=0,ignore_index=True)
    return data[[fea+"_rank"]]

rank_fea = rankFeature("district")
data = pd.read_pickle("../Feature/_featureEnineering_v5.pickle")
print(data.shape)
print(rank_fea.shape)
data = pd.concat([data,rank_fea],axis=1)

train = data[data["month_rent"]!=-1]
test = data[data["month_rent"]==-1]
X = train.drop(["month_rent"],axis=1).values
y = train["month_rent"].values
X_test = test.drop(["month_rent"],axis=1).values
cols = test.drop(["month_rent"],axis=1).columns

print(X.shape,y.shape,X_test.shape)

#---------------------------------------------------------------------
from sklearn.cross_validation import StratifiedKFold
seed_ls = []
# 五折交叉训练，构造五个模型
new_train = None
new_test = copy.deepcopy(test)
skf=list(StratifiedKFold(y, n_folds=5, shuffle=True, random_state=1024))
baseloss = []
loss = 0
new_test["predict"] = 0
for i, (train_index, test_index) in enumerate(skf):
    print("Fold", i)
    model = lgb.LGBMRegressor(objective='regression',num_leaves=125,
                              learning_rate=0.05, n_estimators=2500,boosting_type="gbdt",max_depth=-1,
                             seed=2018,num_thread=-1,max_bin=425,bagging_fraction=0.8,colsample_bytree=0.9,subsample=0.8,lambda_l2=0.20)
    #
    #------------------------------------#
    lgb_model = model.fit(X[train_index], y[train_index],
                          eval_names =['train','valid'],
                          eval_metric= 'rmse',
                          eval_set=[(X[train_index], y[train_index]), 
                                    (X[test_index], y[test_index])],early_stopping_rounds=100)
    baseloss.append(lgb_model.best_score_['valid']['rmse'])
    loss += lgb_model.best_score_['valid']['rmse']
    test_pred= lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration_)
    if i==0:
        predict = test_pred
    else:
        predict = np.vstack((predict,test_pred))
        
    #训练集
    train_predict = lgb_model.predict(X[test_index], num_iteration=lgb_model.best_iteration_)
    train_tmp = train.iloc[test_index]
    train_tmp["predict"] = train_predict
    new_train = pd.concat([new_train,train_tmp],axis=0,ignore_index=False)
    
    #测试集
    new_test["predict"]  = new_test["predict"] + test_pred
new_test["predict"] = new_test["predict"]/5
print('logloss:', baseloss, loss/5)

#----------------------------------------------------------
new_test.to_pickle("../Feature/_featureEnineering_v5.2_stacking1_test.pickle")
new_train.to_pickle("../Feature/_featureEnineering_v5.2_stacking1_train.pickle")

new_test = pd.read_pickle("../Feature/_featureEnineering_v5.2_stacking1_test.pickle")
new_train = pd.read_pickle("../Feature/_featureEnineering_v5.2_stacking1_train.pickle")
new_test = new_test[new_train.columns]

#--------------------------------------------------------
X = new_train.drop(["month_rent"],axis=1).values
y = new_train["month_rent"].values
X_test = new_test.drop(["month_rent"],axis=1).values
cols = new_test.drop(["month_rent"],axis=1).columns

print(X.shape,y.shape,X_test.shape)

print("第二层")
#--------------------------
from sklearn.cross_validation import StratifiedKFold
seed_ls = []
# 五折交叉训练，构造五个模型
skf=list(StratifiedKFold(y, n_folds=5, shuffle=True, random_state=1024))
baseloss = []
loss = 0
for i, (train_index, test_index) in enumerate(skf):
    print("Fold", i)
    model = lgb.LGBMRegressor(objective='regression',num_leaves=125,
                              learning_rate=0.05, n_estimators=2500,boosting_type="gbdt",max_depth=-1,
                             seed=2018,num_thread=-1,max_bin=425,bagging_fraction=0.8,colsample_bytree=0.9,subsample=0.8,lambda_l2=0.20)
    #
    #------------------------------------#
    lgb_model = model.fit(X[train_index], y[train_index],
                          eval_names =['train','valid'],
                          eval_metric= 'rmse',
                          eval_set=[(X[train_index], y[train_index]), 
                                    (X[test_index], y[test_index])],early_stopping_rounds=100)
    baseloss.append(lgb_model.best_score_['valid']['rmse'])
    loss += lgb_model.best_score_['valid']['rmse']
    test_pred= lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration_)
    if i==0:
        predict = test_pred
    else:
        predict = np.vstack((predict,test_pred))
        
print('logloss:', baseloss, loss/5)

submission= pd.DataFrame(predict.mean(axis=0))
test = pd.read_csv("../Data/test.csv")
submission["id"] = test["id"].values
submission.columns=["price","id"]
submission[["id","price"]].to_csv("../Result/_baseline_线下__v5.2_stacking.csv",index=False,encoding="utf-8",sep=",")