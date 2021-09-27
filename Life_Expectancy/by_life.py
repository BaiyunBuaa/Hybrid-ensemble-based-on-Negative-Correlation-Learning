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
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

import multiprocessing
n_jobs = 60

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

#顺序变量做标准化，名义变量不用做标准化
dataset = pd.read_csv('C:/Users/XEON-B00/Desktop/baiyun/Life_Ecpectancy/Life_Expectancy_Data.csv')
dataset = dataset.dropna(axis=0,how='any')
columns = list(dataset.columns)
columns.remove('Life expectancy ')
x = dataset[columns]
y = dataset[['Life expectancy ']]


#处理顺序变量
le = preprocessing.LabelEncoder()
le.fit(np.concatenate([x['Year']]))
x['Year'] = le.transform(x['Year'])

#处理名义变量    
df = pd.get_dummies(x, columns = ['Country','Status'], dummy_na = False)
original_column = list(x.columns)
new_column = [c for c in df.columns if c not in original_column ]

root = 'C:/Users/XEON-B00/Desktop/baiyun/Life_Ecpectancy/pkl_data/'


#数据标准化，先对训练集标准化然后在测试集上应用训练集的规则
x_train, x_test, y_train, y_test = model_selection.train_test_split(df, y,
                                                                    test_size=0.2,
                                                                    random_state=0,
                                                                    shuffle=True)
#with open(root+'x_train.pkl','rb') as file:
#    x_train = pickle.load(file)

scaler_x = StandardScaler()
columns.remove('Country')
columns.remove('Status')
scaler_x.fit(x_train[columns])
x1 = scaler_x.transform(x_train[columns])
#x_train['age'],x_train['bmi'],x_train['children'] = x1[:,0],x1[:,1],x1[:,2]
x2 = scaler_x.transform(x_test[columns])
#x_test['age'],x_test['bmi'],x_test['children'] = x2[:,0],x2[:,1],x2[:,2]

for i in range(len(columns)):
    x_train[columns[i]] = x1[:,i]
    x_test[columns[i]] = x2[:,i]
x_train_sta = x_train
x_test_sta = x_test

scaler_y = StandardScaler()
scaler_y.fit(y_train)
y_train_sta = scaler_y.transform(y_train)
y_test_sta = scaler_y.transform(y_test)

with open(root+'x_train_sta.pkl','wb') as file:
    pickle.dump(x_train_sta,file)
with open(root+'x_test_sta.pkl','wb') as file:
    pickle.dump(x_test_sta,file)
with open(root+'y_train_sta.pkl','wb') as file:
    pickle.dump(y_train_sta,file)
with open(root+'y_test_sta.pkl','wb') as file:
    pickle.dump(y_test_sta,file)

#普通最小二乘
print("==================================")
print("==================================")
print("============1普通最小二乘==============")
print("==================================")
print("==================================")
param_grid_1 = {'n_jobs':[1,-1]}

reg_1 = linear_model.LinearRegression(normalize=False,fit_intercept=False,copy_X=True)
grid_search_1 = GridSearchCV(reg_1, param_grid_1, cv=5,n_jobs=n_jobs,verbose=20)   #网格搜索+交叉验证
grid_search_1.fit(x_train_sta, y_train_sta[:,0])

y_pre_1 = grid_search_1.predict(x_test_sta)
with open(root+'y_pre_1.pkl','wb') as file:
    pickle.dump(y_pre_1,file)

y_pre_1 = y_pre_1.reshape(len(y_pre_1),1)    

f1 = open(root+'y1.txt','a')
l1 = [str(grid_search_1.best_params_)+'\n', str(grid_search_1.score(x_train_sta, y_train_sta))+'\n',
      str(grid_search_1.best_score_)+'\n',str(RMSE(y_test_sta,y_pre_1))+'\n',str(MAE(y_test_sta,y_pre_1))+'\n',
      str(MAPE(y_test_sta,y_pre_1))+'\n']
f1.writelines(l1)
f1.close()

#岭回归
print("==================================")
print("==================================")
print("============2岭回归==============")
print("==================================")
print("==================================")
param_grid_2 = {'alpha': [0.5,1,2],'max_iter':[100,500,1000],
                'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                'tol':[0.0001,0.001,0.01]}
reg_2 = linear_model.Ridge(normalize=False,fit_intercept=False,copy_X=True)
grid_search_2 = GridSearchCV(reg_2, param_grid_2, cv=5, n_jobs=n_jobs,verbose=20)   #网格搜索+交叉验证
x_train_sta = np.ascontiguousarray(x_train_sta)
y_train_sta = np.ascontiguousarray(y_train_sta)
grid_search_2.fit(x_train_sta, y_train_sta[:,0])

y_pre_2 = grid_search_2.predict(x_test_sta)
with open(root+'y_pre_2.pkl','wb') as file:
    pickle.dump(y_pre_2,file)
    
y_pre_2 = y_pre_2.reshape(len(y_pre_2),1)       

f2 = open(root+'y2.txt','a')
l2 = [str(grid_search_2.best_params_)+'\n', str(grid_search_2.score(x_train_sta, y_train_sta))+'\n',
      str(grid_search_2.best_score_)+'\n',str(RMSE(y_test_sta,y_pre_2))+'\n',str(MAE(y_test_sta,y_pre_2))+'\n',
      str(MAPE(y_test_sta,y_pre_2))+'\n']
f2.writelines(l2)
f2.close()

#Lasso回归
print("==================================")
print("==================================")
print("============3Lasso回归==============")
print("==================================")
print("==================================")
param_grid_3 = {'alpha': [0.5,1,2],'precompute':[True,False],
                'max_iter':[100,500,1000],
                'warm_start':[True,False], 'positive':[True,False],
                'selection':['cyclic', 'random'],'tol':[0.0001,0.001,0.01]}
reg_3 = linear_model.Lasso(normalize=False,fit_intercept=False,copy_X=True)
grid_search_3 = GridSearchCV(reg_3, param_grid_3, cv=5, n_jobs=n_jobs,verbose=20)  #网格搜索+交叉验证
grid_search_3.fit(x_train_sta, y_train_sta[:,0])

y_pre_3 = grid_search_3.predict(x_test_sta)
with open(root+'y_pre_3.pkl','wb') as file:
    pickle.dump(y_pre_3,file)

y_pre_3 = y_pre_3.reshape(len(y_pre_3),1)       

f3 = open(root+'y3.txt','a')
l3 = [str(grid_search_3.best_params_)+'\n', str(grid_search_3.score(x_train_sta, y_train_sta))+'\n',
      str(grid_search_3.best_score_)+'\n',str(RMSE(y_test_sta,y_pre_3))+'\n',str(MAE(y_test_sta,y_pre_3))+'\n',
      str(MAPE(y_test_sta,y_pre_3))+'\n']
f3.writelines(l3)
f3.close()


#贝叶斯回归
print("==================================")
print("==================================")
print("============4贝叶斯回归==============")
print("==================================")
print("==================================")
param_grid_4 = {'n_iter':[100,300,500], 'tol':[0.0001,0.001,0.01],
                'alpha_1':[0.000001,0.0001],'alpha_2':[0.000001,0.0001],
                'lambda_1':[0.000001,0.0001],'lambda_2':[0.000001,0.0001],
                'compute_score':[True,False],'fit_intercept':[True,False]
                }
reg_4 = linear_model.BayesianRidge(normalize=False)
grid_search_4 = GridSearchCV(reg_4, param_grid_4, cv=5, n_jobs=n_jobs,verbose=20)   #网格搜索+交叉验证
grid_search_4.fit(x_train_sta, y_train_sta[:,0])

y_pre_4 = grid_search_4.predict(x_test_sta)
with open(root+'y_pre_4.pkl','wb') as file:
    pickle.dump(y_pre_4,file)

y_pre_4 = y_pre_4.reshape(len(y_pre_4),1)     
  
f4 = open(root+'y4.txt','a')
l4 = [str(grid_search_4.best_params_)+'\n', str(grid_search_4.score(x_train_sta, y_train_sta))+'\n',
      str(grid_search_4.best_score_)+'\n',str(RMSE(y_test_sta,y_pre_4))+'\n',str(MAE(y_test_sta,y_pre_4))+'\n',
      str(MAPE(y_test_sta,y_pre_4))+'\n']
f4.writelines(l4)
f4.close()

#随机梯度下降回归
print("==================================")
print("==================================")
print("============5随机梯度下降回归==============")
print("==================================")
print("==================================")
param_grid_5 = {'loss':['squared_loss','huber','epsilon_insensitive','squared_epsilon_insensitive'],
                'penalty':['l1','l2','elasticnet'],'alpha':[0.00001,0.0001,0.001],
                'max_iter':[500,1000,1500],'tol':[0.0001,0.001,0.01],
                'learning_rate':['constant','optimal','invscaling','adaptive']}
reg_5 = linear_model.SGDRegressor()
grid_search_5 = GridSearchCV(reg_5, param_grid_5, cv=5, n_jobs=n_jobs,verbose=20)   #网格搜索+交叉验证
grid_search_5.fit(x_train_sta,y_train_sta[:,0])

y_pre_5 = grid_search_5.predict(x_test_sta)
with open(root+'y_pre_5.pkl','wb') as file:
    pickle.dump(y_pre_5,file)

y_pre_5 = y_pre_5.reshape(len(y_pre_5),1)       
f5 = open(root+'y5.txt','a')
l5 = [str(grid_search_5.best_params_)+'\n', str(grid_search_5.score(x_train_sta, y_train_sta))+'\n',
      str(grid_search_5.best_score_)+'\n',str(RMSE(y_test_sta,y_pre_5))+'\n',str(MAE(y_test_sta,y_pre_5))+'\n',
      str(MAPE(y_test_sta,y_pre_5))+'\n']
f5.writelines(l5)
f5.close()


#随机森林回归
print("==================================")
print("==================================")
print("============7随机森林回归==============")
print("==================================")
print("==================================")
param_grid_7 = {'n_estimators':[50,100,200],'max_depth':[2,3,4],
                'min_samples_split':[2,3,4],'min_samples_leaf':[2,3],
                'bootstrap':[True,False]}
reg_7 = RandomForestRegressor()
grid_search_7 = GridSearchCV(reg_7, param_grid_7, cv=5, n_jobs=n_jobs,verbose=20)
grid_search_7.fit(x_train_sta, y_train_sta[:,0])

y_pre_7 = grid_search_7.predict(x_test_sta)
with open(root+'y_pre_7.pkl','wb') as file:
    pickle.dump(y_pre_7,file)
    
y_pre_7 = y_pre_7.reshape(len(y_pre_7),1)     

f7 = open(root+'y7.txt','a')
l7 = [str(grid_search_7.best_params_)+'\n', str(grid_search_7.score(x_train_sta, y_train_sta))+'\n',
      str(grid_search_7.best_score_)+'\n',str(RMSE(y_test_sta,y_pre_7))+'\n',str(MAE(y_test_sta,y_pre_7))+'\n',
      str(MAPE(y_test_sta,y_pre_7))+'\n']
f7.writelines(l7)
f7.close()



#自适应提升回归
print("==================================")
print("==================================")
print("============8自适应提升回归==============")
print("==================================")
print("==================================")
param_grid_8 = {'n_estimators':[10,50,100],'learning_rate':[0.01,0.1,1],
                'loss':['linear','square','exponential']}
reg_8 = AdaBoostRegressor()
grid_search_8 = GridSearchCV(reg_8, param_grid_8, cv=5, n_jobs=n_jobs,verbose=20)
grid_search_8.fit(x_train_sta, y_train_sta[:,0])

y_pre_8 = grid_search_8.predict(x_test_sta)
with open(root+'y_pre_8.pkl','wb') as file:
    pickle.dump(y_pre_8,file)

y_pre_8 = y_pre_8.reshape(len(y_pre_8),1)  
     
f8 = open(root+'y8.txt','a')
l8 = [str(grid_search_8.best_params_)+'\n', str(grid_search_8.score(x_train_sta, y_train_sta))+'\n',
      str(grid_search_8.best_score_)+'\n',str(RMSE(y_test_sta,y_pre_8))+'\n',str(MAE(y_test_sta,y_pre_8))+'\n',
      str(MAPE(y_test_sta,y_pre_8))+'\n']
f8.writelines(l8)
f8.close()

#梯度提升决策树
print("==================================")
print("==================================")
print("============9梯度提升决策树==============")
print("==================================")
print("==================================")
param_grid_9 = {'n_estimators':[50,100,200],'learning_rate':[0.01,0.1,0.5],
                'loss':['ls','lad','huber','quantile'],'min_samples_split':[2,3],
                'max_depth':[2,3,4]}
reg_9 = GradientBoostingRegressor()
grid_search_9 = GridSearchCV(reg_9, param_grid_9, cv=5, n_jobs=n_jobs,verbose=20)
grid_search_9.fit(x_train_sta, y_train_sta[:,0])

y_pre_9 = grid_search_9.predict(x_test_sta)
with open(root+'y_pre_9.pkl','wb') as file:
    pickle.dump(y_pre_9,file)

y_pre_9 = y_pre_9.reshape(len(y_pre_9),1)           

f9 = open(root+'y9.txt','a')
l9 = [str(grid_search_9.best_params_)+'\n', str(grid_search_9.score(x_train_sta, y_train_sta))+'\n',
      str(grid_search_9.best_score_)+'\n',str(RMSE(y_test_sta,y_pre_9))+'\n',str(MAE(y_test_sta,y_pre_9))+'\n',
      str(MAPE(y_test_sta,y_pre_9))+'\n']
f9.writelines(l9)
f9.close()

#支持向量回归
print("==================================")
print("==================================")
print("============10支持向量回归==============")
print("==================================")
print("==================================")
param_grid_10 = {'kernel':['linear','poly','rbf'],
                  'degree':[2,3,4],'C':[0.5,1,2],'gamma':['scale','auto']}
reg_10 = SVR()
grid_search_10 = GridSearchCV(reg_10, param_grid_10, cv=5, n_jobs=n_jobs,verbose=20)
grid_search_10.fit(x_train_sta, y_train_sta[:,0])

y_pre_10 = grid_search_10.predict(x_test_sta)
with open(root+'y_pre_10.pkl','wb') as file:
    pickle.dump(y_pre_10,file)

y_pre_10 = y_pre_10.reshape(len(y_pre_10),1)       

f10 = open(root+'y10.txt','a')
l10 = [str(grid_search_10.best_params_)+'\n', str(grid_search_10.score(x_train_sta, y_train_sta))+'\n',
      str(grid_search_10.best_score_)+'\n',str(RMSE(y_test_sta,y_pre_10))+'\n',str(MAE(y_test_sta,y_pre_10))+'\n',
      str(MAPE(y_test_sta,y_pre_10))+'\n']
f10.writelines(l10)
f10.close()

#决策树回归
print("==================================")
print("==================================")
print("============11决策树回归==============")
print("==================================")
print("==================================")
param_grid_11 = {'splitter':['best','random'], 'min_samples_split':[2,3],
                  'min_samples_leaf':[2,3],'max_features':['auto','sqrt','log2']}
reg_11 = DecisionTreeRegressor()
grid_search_11 = GridSearchCV(reg_11, param_grid_11, cv=5, n_jobs=n_jobs,verbose=20)
grid_search_11.fit(x_train_sta, y_train_sta[:,0])

y_pre_11 = grid_search_11.predict(x_test_sta)
with open(root+'y_pre_11.pkl','wb') as file:
    pickle.dump(y_pre_11,file)

y_pre_11 = y_pre_11.reshape(len(y_pre_11),1)       

f11 = open(root+'y11.txt','a')
l11 = [str(grid_search_11.best_params_)+'\n', str(grid_search_11.score(x_train_sta, y_train_sta))+'\n',
      str(grid_search_11.best_score_)+'\n',str(RMSE(y_test_sta,y_pre_11))+'\n',str(MAE(y_test_sta,y_pre_11))+'\n',
      str(MAPE(y_test_sta,y_pre_11))+'\n']
f11.writelines(l11)
f11.close()

#多层感知机回归
print("==================================")
print("==================================")
print("============12多层感知机回归==============")
print("==================================")
print("==================================")
param_grid_12 = {'activation':['identity','logistic','tanh','relu'], 'solver':['sgd','adam'],
                  'alpha':[0.00001,0.0001,0.001],'learning_rate':['constant','invscaling','adaptive']}
reg_12 = MLPRegressor()
grid_search_12 = GridSearchCV(reg_12, param_grid_12, cv=5, n_jobs=n_jobs,verbose=20)
grid_search_12.fit(x_train_sta, y_train_sta[:,0])

y_pre_12 = grid_search_12.predict(x_test_sta)
with open(root+'y_pre_12.pkl','wb') as file:
    pickle.dump(y_pre_12,file)

y_pre_12 = y_pre_12.reshape(len(y_pre_12),1)       

f12 = open(root+'y12.txt','a')
l12 = [str(grid_search_12.best_params_)+'\n', str(grid_search_12.score(x_train_sta, y_train_sta))+'\n',
      str(grid_search_12.best_score_)+'\n',str(RMSE(y_test_sta,y_pre_12))+'\n',str(MAE(y_test_sta,y_pre_12))+'\n',
      str(MAPE(y_test_sta,y_pre_12))+'\n']
f12.writelines(l12)
f12.close()

#多项式回归
print("==================================")
print("==================================")
print("============6多项式回归==============")
print("==================================")
print("==================================")
def PolynomialRegression():
    return make_pipeline(PolynomialFeatures(),
                          linear_model.LinearRegression(normalize=False,fit_intercept=False,copy_X=True))
    
param_grid_6 = {'polynomialfeatures__degree':[2,3], 'polynomialfeatures__interaction_only':[True,False],
                'polynomialfeatures__include_bias':[True,False],'polynomialfeatures__order':['C','F']}
grid_search_6 = GridSearchCV(PolynomialRegression(), param_grid_6, cv=5, n_jobs=n_jobs,verbose=20)
grid_search_6.fit(x_train_sta, y_train_sta[:,0])

y_pre_6 = grid_search_6.predict(x_test_sta)
with open(root+'y_pre_6.pkl','wb') as file:
    pickle.dump(y_pre_6,file)

y_pre_6 = y_pre_6.reshape(len(y_pre_6),1)        

f6 = open(root+'y6.txt','a')
l6 = [str(grid_search_6.best_params_)+'\n', str(grid_search_6.score(x_train_sta, y_train_sta))+'\n',
      str(grid_search_6.best_score_)+'\n',str(RMSE(y_test_sta,y_pre_6))+'\n',str(MAE(y_test_sta,y_pre_6))+'\n',
      str(MAPE(y_test_sta,y_pre_6))+'\n']
f6.writelines(l6)
f6.close()












