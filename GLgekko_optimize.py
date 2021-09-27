# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 16:51:46 2020

@author: nihao
"""

import numpy as np
import pickle
from gekko import GEKKO
import math
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch.utils.data as Data

class ConvModel_8(nn.Module):
    def __init__(self,pool_num):
        super(ConvModel_8,self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size=2)
        self.maxpool = nn.MaxPool1d(kernel_size=2) 
        self.fc1 = nn.Linear(pool_num, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def loadData():
    root = './House/pkl_data/'
    with open(root+'y_pre_1.pkl','rb') as file:
        temp = pickle.load(file)
    print('size=',len(temp))    
    result_array = np.zeros([len(temp),12])
    del temp
    
    for i in range(1,13):
        with open(root+'y_pre_'+str(i)+'.pkl','rb') as file:
            temp = pickle.load(file)
            result_array[:,i-1] = temp.reshape(-1,)
    
    with open(root+'y_test_sta.pkl','rb') as file:
        y_test_real = pickle.load(file)
        
    size = 0.3
    select_array = result_array[:int(len(result_array)*size)]
    test_array = result_array[int(len(result_array)*size):]
    
    select_y_real = y_test_real[:int(len(y_test_real)*size)]
    test_y_real = y_test_real[int(len(y_test_real)*size):]
    
    return select_array,select_y_real,test_array,test_y_real

def RMSE(l1,l2):
    n = len(l1)
    rmse = math.sqrt(sum(np.square(l1-l2))/n)
    return rmse

def MAE(l1,l2):
    n = len(l1)
    mae = sum(np.abs(l1-l2))/n
    return mae[0]

def MAPE(l1,l2):
    n = len(l1)
    mape = sum(np.abs((l1-l2)/l1))/n
    return mape[0] 
    
def fitness_func(lambda_,x,m):
    for i in range(select_array.shape[1]):
        for j in range(select_array.shape[0]):
            m.Minimize(((select_array[j][i] - select_y_real[j][0])**2*x[i]-(select_array[j][i] - 
                        select_array[j][i]*x[i]/sum(x))**2*x[i]*lambda_))    

def get_solution(lambda_):
    m = GEKKO(remote=False)
    x = m.Array(m.Var,12,lb=0,ub=1)    

    fitness_func(lambda_,x,m)
    m.Equation(m.sum(x)==1)
    m.options.SOLVER=3 #IPOPT

    m.solve(disp=False)
    print(x)

    solution = []
    for i in range(len(x)):
        solution.append(x[i][0])
        
    return solution
    
def compute_Z(lambda_):
    solution = get_solution(lambda_)
    avg_weight = [solution[i]/sum(solution) for i in range(len(solution))]
    select_avg = np.average(select_array,weights=avg_weight,axis=1)
    rmse = RMSE(select_y_real,select_avg.reshape(-1,1))
    mae = MAE(select_y_real,select_avg.reshape(-1,1))
    mape = MAPE(select_y_real,select_avg.reshape(-1,1))
    
    print('rmse', rmse)
    print('mae',mae)
    print('mape',mape)
    return((rmse+mae+mape)/3)

def findLambda():
    step = 0.1
    start_lambda = 0.1
    while True:
        lambda_list = [start_lambda+i*step for i in range(10)] + [start_lambda-i*step for i in range(1,10)]
        Z = compute_Z(start_lambda)
        best_lambda = 0.1
        for lambda_ in lambda_list:
            print(lambda_)
            if lambda_ >= 0 and lambda_ <= 1:
                if compute_Z(lambda_) < Z:
                    Z = compute_Z(lambda_)
                    best_lambda = lambda_
        step = step/10
        start_lambda = best_lambda
        if step < 0.001:
            return Z,best_lambda

def compute_metrics(test_y_real,new_array):
    print('rmse', RMSE(test_y_real,new_array.reshape(-1,1)))
    print('mae', MAE(test_y_real,new_array.reshape(-1,1)))
    print('mape', MAPE(test_y_real,new_array.reshape(-1,1)))

def Weight_Finetune():
    select_matric_list = []
    for i in range(12):
        rmse = RMSE(select_y_real,select_array[:,i].reshape(-1,1))
        mae = MAE(select_y_real,select_array[:,i].reshape(-1,1))
        mape = MAPE(select_y_real,select_array[:,i].reshape(-1,1))
        select_matric_list.append((rmse+mae+mape)/3)
#        select_matric_list.append(rmse)
    weight_inv = [1/select_matric_list[i] for i in range(len(select_matric_list))]
    weight_inv = [weight_inv[i]/sum(weight_inv) for i in range(len(weight_inv))]
    weight_inv = np.array(weight_inv).reshape(12,)
    weight_exp = [math.exp(-select_matric_list[i]) for i in range(len(select_matric_list))]
    weight_exp = [weight_exp[i]/sum(weight_exp) for i in range(len(weight_exp))]
    weight_exp = np.array(weight_exp).reshape(12,)

    return weight_inv,weight_exp
        
def Single(weight_inv,weight_exp):
    print('#'*40)
    print('#'*20+'Single model'+'#'*20)
    print('#'*40)
    test_matric_list = []
#    modelList = ['CNN1','CNN2','CNN3','CNN4','CNN5','CNN6','CNN7','CNN8',
#                 'CNN9','CNN10']
    modelList = ['SLR','RR','LR','BR','SGDR','PR','RFR',
                 'ABR','GBDT','SVR','DTR','MLP']
    for i in range(12):
        rmse = RMSE(test_y_real,test_array[:,i].reshape(-1,1))
        mae = MAE(test_y_real,test_array[:,i].reshape(-1,1))
        mape = MAPE(test_y_real,test_array[:,i].reshape(-1,1))
        test_matric_list.append((rmse+mae+mape)/3)
        print('sub-model:',modelList[i])
        print('rmse:',rmse)
        print('mae:',mae)
        print('mape:',mape)
#        test_matric_list.append(rmse)
    
    model_index = test_matric_list.index(min(test_matric_list))   
    print('The best model is: ',modelList[model_index])
    compute_metrics(test_y_real,test_array[:,model_index])
    
    
    print('#'*20+'Simple average'+'#'*20)
    bar_array = np.mean(test_array,axis = 1)
    compute_metrics(test_y_real,bar_array)
    
    print('#'*20+'Simple average and Weighted average-inv'+'#'*20)
    y_inv = np.average(test_array,weights = weight_inv,axis = 1)
    compute_metrics(test_y_real,y_inv)  
    
    print('#'*20+'Simple average and Weighted average-exp'+'#'*20)
    y_exp = np.average(test_array,weights = weight_exp,axis = 1)
    compute_metrics(test_y_real,y_exp)  

def Combination(solution,weight_inv,weight_exp):
    print('#'*40)
    print('#'*20+'Hybird ensemble'+'#'*20)
    print('#'*40)
          
    print('#'*20+'NCL组合模型'+'#'*20)
    test_ncl = np.average(test_array,weights=solution,axis=1)
    compute_metrics(test_y_real,test_ncl)
    
    print('#'*20+'NCL-inv-Hybird ensemble'+'#'*20)
    solution_inv = list(np.array(solution)*np.array(weight_inv))    
    weight_solution_inv = [solution_inv[i]/sum(solution_inv) for i in range(len(solution_inv))] 
    weight_solution_inv = np.array(weight_solution_inv).reshape(12,)
    test_ncl_inv = np.average(test_array,weights=weight_solution_inv,axis=1)
    compute_metrics(test_y_real,test_ncl_inv)
    
    print('#'*20+'NCL-exp-Hybird ensemble'+'#'*20)
    solution_exp = list(np.array(solution)*np.array(weight_exp))
    weight_solution_exp = [solution_exp[i]/sum(solution_exp) for i in range(len(solution_exp))] 
    weight_solution_exp = np.array(weight_solution_exp).reshape(12,)
    test_ncl_exp = np.average(test_array,weights=weight_solution_exp,axis=1)
    compute_metrics(test_y_real,test_ncl_exp)

def as_num(L):
    L = [float('{:.6f}'.format(x)) for x in L[0]]
    return(L)
    
def LR_combination(x_train,x_test,y_train,y_test):
    reg = linear_model.LinearRegression(normalize=False,fit_intercept=False,copy_X=True)
    reg.fit(x_train, y_train) 
    print('LR parameters',as_num(reg.coef_))
    y_pre = reg.predict(x_test)    
    compute_metrics(y_test,y_pre)

def CNN8_train(x_train,y_train,convmodel):
    train_data = Data.TensorDataset(x_train,y_train)
    train_loader = Data.DataLoader(dataset=train_data, 
                                   batch_size=512,
                                   shuffle=True,
                                   num_workers=0)

    model = convmodel
    optimizer = opt.SGD(model.parameters(),lr=0.001) 
    loss_function = nn.MSELoss()   
    
    train_loss_all = [] 
    
    # Train
    channel = x_train.size()[1]
    for epoch in range(200): 
        for step, (b_x, b_y) in enumerate(train_loader):
            if b_x.size()[0] < 512:
                break
            b_x = b_x.unsqueeze(2)
            b_x = b_x.reshape([512,1,channel])
            output = model(b_x)
            train_loss = loss_function(output.squeeze(1), b_y) 
            optimizer.zero_grad() 
            train_loss.backward()
            optimizer.step() 
            train_loss_all.append(train_loss.item())
        print(epoch)
    return model   
 
def CNN8_combination(x_train,x_test,y_train,y_test):
    x_train = torch.from_numpy(x_train.astype(np.float32))                      
    x_test = torch.from_numpy(x_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))                       
#    y_test = torch.from_numpy(y_test.astype(np.float32)) 
    
    pool_num_8 = math.floor((x_train.size()[1]-(2-1))/2)
    convmodel = ConvModel_8(pool_num_8)
    
    conv_model = CNN8_train(x_train,y_train,convmodel)
    y_pred = conv_model(x_test.unsqueeze(1)).squeeze(1).squeeze(1)
    y_pred = y_pred.detach().numpy().reshape(-1,1)

    compute_metrics(y_test,y_pred)
    
def Bagging_combination(X_train, X_test, y_train, y_test):
    
    model_car = GradientBoostingRegressor(learning_rate= 0.1, loss= 'huber', 
                                          max_depth = 4, min_samples_split = 3, 
                                          n_estimators = 200)
    
    model_life = SVR(C = 2, degree = 2, gamma = 'scale', kernel = 'linear')
    
    model_walmart = DecisionTreeRegressor(max_features = 'auto', min_samples_leaf = 3, 
                                          min_samples_split = 2, splitter = 'random')
    
    model_house = GradientBoostingRegressor(learning_rate = 0.1, loss = 'huber', 
                                            max_depth = 4, min_samples_split = 2, 
                                            n_estimators = 200)
    
    model_insurance = SVR(C = 2, degree = 2, gamma = 'auto', kernel = 'rbf')
    
    
##    x_train_sta = np.ascontiguousarray(X_train)
##    y_train_sta = np.ascontiguousarray(y_train)
##    print(X_train.flags)
#    X_train = X_train.copy(order='F')
#    print(X_train.flags)
#    y_train = y_train.copy(order='F')
    
    model_black = linear_model.Ridge(alpha = 0.5, copy_X = True, 
                                     fit_intercept = False, max_iter = 2000, 
                                     solver = 'sag', tol = 0.001,
                                     normalize=False)
      
# If  you want you want to use another models just replace the model_black in base_estimator with others
    black_bagging = BaggingRegressor(base_estimator = model_house,
                                     random_state = 0).fit(X_train, y_train)
    
    y_pred = black_bagging.predict(X_test)
    compute_metrics(y_test,y_pred)

#This function is used to compute the diversity in an ensemble
def div1(test_array):
    M = test_array.shape[1]
    N = test_array.shape[0]
    avg_test = np.average(test_array,axis=1).reshape(-1,1)
    expand = np.expand_dims(avg_test,1).repeat(M,axis=1).reshape(N,M)
    Sum = np.sum((test_array-expand)**2)
    return Sum/(M*N)             

if __name__ == "__main__":
    select_array,select_y_real,test_array,test_y_real = loadData()
    
# If you use the Life_Expectancy dataset, remember add the following sentence, becuse there is a super large value
#    test_array[:,0][44],test_array[:,0][134] = 0,0  
    print(div1(test_array))
    
    weight_inv,weight_exp = Weight_Finetune()
    
    Z,best_lambda = findLambda()
    print('best_lambda',best_lambda)
    solution = get_solution(best_lambda)

    # Single model
    Single(weight_inv,weight_exp)
    
    # Hybrid ensemble
    Combination(solution,weight_inv,weight_exp)
    
    # Other benchmarks
    LR_combination(select_array,test_array,select_y_real,test_y_real)
    CNN8_combination(select_array,test_array,select_y_real,test_y_real)
    Bagging_combination(select_array,test_array,select_y_real,test_y_real)
