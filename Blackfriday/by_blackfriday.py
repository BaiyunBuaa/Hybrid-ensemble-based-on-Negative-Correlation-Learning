# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:14:40 2020

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
from sklearn.pipeline import make_pipeline

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

# #顺序变量做标准化，名义变量不用做标准化
# dataset = pd.read_csv('D:/ByResearch/negtive-correlation-learning/data/Regression_dataset/BlackFriday/BlackFriday.csv')

# x = dataset[['Gender', 'Age', 'Occupation', 'City_Category',
#        'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1']]
# y = dataset[['Purchase']]


# #处理顺序变量
# for col in ['Age','Stay_In_Current_City_Years']:
#     le = preprocessing.LabelEncoder()
#     le.fit(np.concatenate([x[col]]))
#     x[col] = le.transform(x[col])

# #处理名义变量    
# df = pd.get_dummies(x, columns = ['Gender','Occupation', 'City_Category',
#                                    'Marital_Status', 'Product_Category_1'], dummy_na = False)
# original_column = list(x.columns)
# new_column = [c for c in df.columns if c not in original_column ]

# root = 'D:/ByResearch/negtive-correlation-learning/data/Regression_dataset/BlackFriday/pkl_data/'


# #数据标准化，先对训练集标准化然后在测试集上应用训练集的规则
# x_train, x_test, y_train, y_test = model_selection.train_test_split(df, y,
#                                                                     test_size=0.2,
#                                                                     random_state=0,
#                                                                     shuffle=True)
# #with open(root+'x_train.pkl','rb') as file:
# #    x_train = pickle.load(file)

# scaler_x = StandardScaler()
# scaler_x.fit(x_train[['Age','Stay_In_Current_City_Years']])
# x1 = scaler_x.transform(x_train[['Age','Stay_In_Current_City_Years']])
# x_train['Age'],x_train['Stay_In_Current_City_Years'] = x1[:,0],x1[:,1]
# x_train_sta = x_train
# x2 = scaler_x.transform(x_test[['Age','Stay_In_Current_City_Years']])
# x_test['Age'],x_test['Stay_In_Current_City_Years'] = x2[:,0],x2[:,1]
# x_test_sta = x_test

# scaler_y = StandardScaler()
# scaler_y.fit(y_train)
# y_train_sta = scaler_y.transform(y_train)
# y_test_sta = scaler_y.transform(y_test)

# #with open(root+'x_train_sta.pkl','wb') as file:
# #    pickle.dump(x_train_sta,file)


# #普通最小二乘
# param_grid_1 = {'n_jobs':[1,-1]}
# reg_1 = linear_model.LinearRegression(normalize=False,fit_intercept=False,copy_X=True)
# grid_search_1 = GridSearchCV(reg_1, param_grid_1, cv=5)   #网格搜索+交叉验证
# grid_search_1.fit(x_train_sta, y_train_sta)
# print('GridSearchCV交叉验证网格搜索字典获得的最好参数组合',grid_search_1.best_params_)
# print('GridSearchCV交叉验证网格搜索获得的最好估计器,在训练验证集上没做交叉验证的得分',grid_search_1.score(x_train_sta, y_train_sta))
# print('GridSearchCV交叉验证网格搜索获得的最好估计器,在**集上做交叉验证的平均得分',grid_search_1.best_score_)  

# y_pre_1 = grid_search_1.predict(x_test_sta)
# with open(root+'y_pre_1.pkl','wb') as file:
#     pickle.dump(y_pre_1,file)

# RMSE(y_test_sta,y_pre_1)
# MAE(y_test_sta,y_pre_1)
# MAPE(y_test_sta,y_pre_1)

# #岭回归
# param_grid_2 = {'alpha': [0.5,1,2],'max_iter':[100,500,1000],
#                 'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
#                 'tol':[0.0001,0.001,0.01]}
# reg_2 = linear_model.Ridge(normalize=False,fit_intercept=False,copy_X=True)
# grid_search_2 = GridSearchCV(reg_2, param_grid_2, cv=5)   #网格搜索+交叉验证
# x_train_sta = np.ascontiguousarray(x_train_sta)
# y_train_sta = np.ascontiguousarray(y_train_sta)
# grid_search_2.fit(x_train_sta, y_train_sta)

# print('GridSearchCV交叉验证网格搜索字典获得的最好参数组合',grid_search_2.best_params_)
# print('GridSearchCV交叉验证网格搜索获得的最好估计器,在训练验证集上没做交叉验证的得分',grid_search_2.score(x_train_sta, y_train_sta))
# print('GridSearchCV交叉验证网格搜索获得的最好估计器,在**集上做交叉验证的平均得分',grid_search_2.best_score_)  

# y_pre_2 = grid_search_2.predict(x_test_sta)
# with open(root+'y_pre_2.pkl','wb') as file:
#     pickle.dump(y_pre_2,file)
    
# RMSE(y_test_sta,y_pre_2)
# MAE(y_test_sta,y_pre_2)
# MAPE(y_test_sta,y_pre_2)

# #Lasso回归
# param_grid_3 = {'alpha': [0.5,1,2],'precompute':[True,False],
#                 'max_iter':[100,500,1000],
#                 'warm_start':[True,False], 'positive':[True,False],
#                 'selection':['cyclic', 'random'],'tol':[0.0001,0.001,0.01]}
# reg_3 = linear_model.Lasso(normalize=False,fit_intercept=False,copy_X=True)
# grid_search_3 = GridSearchCV(reg_3, param_grid_3, cv=5)   #网格搜索+交叉验证
# grid_search_3.fit(x_train_sta, y_train_sta)

# print('GridSearchCV交叉验证网格搜索字典获得的最好参数组合',grid_search_3.best_params_)
# print('GridSearchCV交叉验证网格搜索获得的最好估计器,在训练验证集上没做交叉验证的得分',grid_search_3.score(x_train_sta, y_train_sta))
# print('GridSearchCV交叉验证网格搜索获得的最好估计器,在**集上做交叉验证的平均得分',grid_search_3.best_score_)  

# y_pre_3 = grid_search_3.predict(x_test_sta)
# with open(root+'y_pre_3.pkl','wb') as file:
#     pickle.dump(y_pre_3,file)

# y_pre_3 = y_pre_3.reshape(len(y_pre_3),1)   
# RMSE(y_test_sta,y_pre_3)
# MAE(y_test_sta,y_pre_3)
# MAPE(y_test_sta,y_pre_3)


# #贝叶斯回归
# param_grid_4 = {'n_iter':[100,300,500], 'tol':[0.0001,0.001,0.01],
#                 'alpha_1':[0.000001,0.0001],'alpha_2':[0.000001,0.0001],
#                 'lambda_1':[0.000001,0.0001],'lambda_2':[0.000001,0.0001],
#                 'compute_score':[True,False],'fit_intercept':[True,False]
#                 }
# reg_4 = linear_model.BayesianRidge(normalize=False)
# grid_search_4 = GridSearchCV(reg_4, param_grid_4, cv=5)   #网格搜索+交叉验证
# grid_search_4.fit(x_train_sta, y_train_sta)

# print('GridSearchCV交叉验证网格搜索字典获得的最好参数组合',grid_search_4.best_params_)
# print('GridSearchCV交叉验证网格搜索获得的最好估计器,在训练验证集上没做交叉验证的得分',grid_search_4.score(x_train_sta, y_train_sta))
# print('GridSearchCV交叉验证网格搜索获得的最好估计器,在**集上做交叉验证的平均得分',grid_search_4.best_score_)  

# y_pre_4 = grid_search_4.predict(x_test_sta)
# with open(root+'y_pre_4.pkl','wb') as file:
#     pickle.dump(y_pre_4,file)

# y_pre_4 = y_pre_4.reshape(len(y_pre_4),1)    
# RMSE(y_test_sta,y_pre_4)
# MAE(y_test_sta,y_pre_4)
# MAPE(y_test_sta,y_pre_4)

# #随机梯度下降回归
root ='./'
with open(root+'x_train_sta.pkl','rb') as file:
    x_train_sta = pickle.load(file)
    
with open(root+'x_test_sta.pkl','rb') as file:
    x_test_sta = pickle.load(file)
    
with open(root+'y_train_sta.pkl','rb') as file:
    y_train_sta = pickle.load(file)

with open(root+'y_test_sta.pkl','rb') as file:
    y_test_sta = pickle.load(file)
    
# param_grid_5 = {'loss':['squared_loss','huber','epsilon_insensitive','squared_epsilon_insensitive'],
#                 'penalty':['l1','l2','elasticnet'],'alpha':[0.00001,0.0001,0.001],
#                 'max_iter':[500,1000,1500],'tol':[0.0001,0.001,0.01],
#                 'learning_rate':['constant','optimal','invscaling','adaptive']}
# reg_5 = linear_model.SGDRegressor()
# grid_search_5 = GridSearchCV(reg_5, param_grid_5, cv=5)   #网格搜索+交叉验证
# grid_search_5.fit(x_train_sta, y_train_sta)

# print('GridSearchCV交叉验证网格搜索字典获得的最好参数组合',grid_search_5.best_params_)
# print('GridSearchCV交叉验证网格搜索获得的最好估计器,在训练验证集上没做交叉验证的得分',grid_search_5.score(x_train_sta, y_train_sta))
# print('GridSearchCV交叉验证网格搜索获得的最好估计器,在**集上做交叉验证的平均得分',grid_search_5.best_score_)  

# y_pre_5 = grid_search_5.predict(x_test_sta)
# with open(root+'y_pre_5.pkl','wb') as file:
#     pickle.dump(y_pre_5,file)

# y_pre_5 = y_pre_5.reshape(len(y_pre_5),1)    
# RMSE(y_test_sta,y_pre_5)
# MAE(y_test_sta,y_pre_5)
# MAPE(y_test_sta,y_pre_5)

#多项式回归
def PolynomialRegression():
    return make_pipeline(PolynomialFeatures(),
                         linear_model.LinearRegression(normalize=False,fit_intercept=False,copy_X=True))
    
param_grid_6 = {'polynomialfeatures__degree':[2,3], 'polynomialfeatures__interaction_only':[True,False],
                'polynomialfeatures__include_bias':[True,False],'polynomialfeatures__order':['C','F']}
grid_search_6 = GridSearchCV(PolynomialRegression(), param_grid_6, cv=5)
grid_search_6.fit(x_train_sta, y_train_sta)

print('GridSearchCV交叉验证网格搜索字典获得的最好参数组合',grid_search_6.best_params_)
print('GridSearchCV交叉验证网格搜索获得的最好估计器,在训练验证集上没做交叉验证的得分',grid_search_6.score(x_train_sta, y_train_sta))
print('GridSearchCV交叉验证网格搜索获得的最好估计器,在**集上做交叉验证的平均得分',grid_search_6.best_score_)  

y_pre_6 = grid_search_6.predict(x_test_sta)
with open(root+'y_pre_6.pkl','wb') as file:
    pickle.dump(y_pre_6,file)

y_pre_6 = y_pre_6.reshape(len(y_pre_6),1)    
# RMSE(y_test_sta,y_pre_6)
# MAE(y_test_sta,y_pre_6)
# MAPE(y_test_sta,y_pre_6)
print('RMSE',RMSE(y_test_sta,y_pre_6))
print('MAE',MAE(y_test_sta,y_pre_6))
print('MAPE',MAPE(y_test_sta,y_pre_6))


# #随机森林回归
# param_grid_7 = {'n_estimators':[50,100,200],'max_depth':[2,3,4],
#                 'min_samples_split':[2,3,4],'min_samples_leaf':[1,2,3],
#                 'bootstrap':[True,False]}
# reg_7 = RandomForestRegressor()
# grid_search_7 = GridSearchCV(reg_7, param_grid_7, cv=5)
# grid_search_7.fit(x_train_sta, y_train_sta)

# print('GridSearchCV交叉验证网格搜索字典获得的最好参数组合',grid_search_7.best_params_)
# print('GridSearchCV交叉验证网格搜索获得的最好估计器,在训练验证集上没做交叉验证的得分',grid_search_7.score(x_train_sta, y_train_sta))
# print('GridSearchCV交叉验证网格搜索获得的最好估计器,在**集上做交叉验证的平均得分',grid_search_7.best_score_)  

# y_pre_7 = grid_search_7.predict(x_test_sta)
# with open(root+'y_pre_7.pkl','wb') as file:
#     pickle.dump(y_pre_7,file)
    
# RMSE(y_test_sta,y_pre_7)
# MAE(y_test_sta,y_pre_7)
# MAPE(y_test_sta,y_pre_7)

# #自适应提升回归
# param_grid_8 = {'n_estimators':[50,100,200],'max_depth':[2,3,4],
#                 'min_samples_split':[2,3,4],'min_samples_leaf':[1,2,3],
#                 'bootstrap':[True,False]}
# reg_7 = RandomForestRegressor()
# grid_search_7 = GridSearchCV(reg_7, param_grid_7, cv=5)
# grid_search_7.fit(x_train_sta, y_train_sta)

# print('GridSearchCV交叉验证网格搜索字典获得的最好参数组合',grid_search_7.best_params_)
# print('GridSearchCV交叉验证网格搜索获得的最好估计器,在训练验证集上没做交叉验证的得分',grid_search_7.score(x_train_sta, y_train_sta))
# print('GridSearchCV交叉验证网格搜索获得的最好估计器,在**集上做交叉验证的平均得分',grid_search_7.best_score_)  

# y_pre_7 = grid_search_7.predict(x_test_sta)
# with open(root+'y_pre_7.pkl','wb') as file:
#     pickle.dump(y_pre_7,file)
    
# RMSE(y_test_sta,y_pre_7)
# MAE(y_test_sta,y_pre_7)
# MAPE(y_test_sta,y_pre_7)












