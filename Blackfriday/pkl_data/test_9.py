# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:57:43 2020

@author: nihao
"""

import pickle
import pandas as pd
import  numpy as np
import math
from sklearn import preprocessing, model_selection,linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

import multiprocessing
print("multiprocessing.cpu_count()",multiprocessing.cpu_count())
n_jobs = multiprocessing.cpu_count()-1

# l1-true,l2-false
def RMSE(l1,l2):
    length = len(l1)
    sum = 0
    for i in range(length):
        sum = sum + np.square(l1[i]-l2[i])
    return math.sqrt(sum/length)
#define MAE
def MAE(l1,l2):
    n = len(l1)
    l1 = np.array(l1)
    l2 = np.array(l2)
    mae = sum(np.abs(l1-l2))/n
    return mae
#def MAPE
def MAPE(l1,l2):
    n = len(l1)
    l1 = np.array(l1)
    l2 = np.array(l2)
    for i in range(len(l1)):
        if l1[i] == 0:
            l1[i] = 0.01
    mape = sum(np.abs((l1-l2)/l1))/n
    return mape


#root = 'D:/ByResearch/negtive-correlation-learning/data/Regression_dataset/BlackFriday/pkl_data/'
root="./"

with open(root+'x_train_sta.pkl','rb') as file:
    x_train_sta = pickle.load(file)
    
with open(root+'x_test_sta.pkl','rb') as file:
    x_test_sta = pickle.load(file)
    
with open(root+'y_train_sta.pkl','rb') as file:
    y_train_sta = pickle.load(file)

with open(root+'y_test_sta.pkl','rb') as file:
    y_test_sta = pickle.load(file)
    
param_grid_9 = {'n_estimators':[50,100,200],'learning_rate':[0.01,0.1,0.5],
                'loss':['ls','lad','huber','quantile'],'min_samples_split':[2,3],
                'max_depth':[2,3,4]}
reg_9 = GradientBoostingRegressor()
grid_search_9 = GridSearchCV(reg_9, param_grid_9, cv=5, n_jobs=n_jobs)
grid_search_9.fit(x_train_sta, y_train_sta[:,0])

print('GridSearchCV交叉验证网格搜索字典获得的最好参数组合',grid_search_9.best_params_)
print('GridSearchCV交叉验证网格搜索获得的最好估计器,在训练验证集上没做交叉验证的得分',grid_search_9.score(x_train_sta, y_train_sta))
print('GridSearchCV交叉验证网格搜索获得的最好估计器,在**集上做交叉验证的平均得分',grid_search_9.best_score_)  

y_pre_9 = grid_search_9.predict(x_test_sta)
with open(root+'y_pre_9.pkl','wb') as file:
    pickle.dump(y_pre_9,file)

y_pre_9 = y_pre_9.reshape(len(y_pre_9),1)      
print('RMSE',RMSE(y_test_sta,y_pre_9))
print('MAE',MAE(y_test_sta,y_pre_9))
print('MAPE',MAPE(y_test_sta,y_pre_9))
