#!/usr/bin/env python
# coding: utf-8


#Load packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import sqrt
from numpy.linalg import inv


#z_score normalization functions


# In[43]:


def normal_train_data(DF):
    Mean = np.mean(DF,axis = 0)
    STD = np.std(DF,axis = 0)
    stdDF = (DF-Mean)/STD
    stdDF['x0']= 1 
    cols = stdDF.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    stdDF = stdDF[cols]
    return stdDF


# In[44]:


def normal_test_data(trainDF,testDF):
    Mean = np.mean(trainDF,axis = 0)
    STD = np.std(trainDF,axis = 0)
    stdDF = (testDF-Mean)/STD
    stdDF['x0']= 1 
    cols = stdDF.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    stdDF = stdDF[cols]
    return stdDF


# dataset 1

#housing data
house = pd.read_csv('housing.csv')

#Split the data and get the training and testing data 
train=house.sample(frac=0.8,random_state=500) 
test=house.drop(train.index)

#normalize the dataset 
normal_train = normal_train_data(train)
normal_test = normal_test_data(train,test)

#get the x features and y 
x_house_train = np.array(normal_train.drop(['Y'], axis=1))
y_house_train = np.array(normal_train[['Y']])

x_house_test = np.array(normal_test.drop(['Y'], axis=1))
y_house_test = np.array(normal_test[['Y']])


#dataset 2

#yacht data
yacht = pd.read_csv('yachtData.csv')

#Split the data and get the training and testing data 
train=yacht.sample(frac=0.8,random_state=500) 
test=yacht.drop(train.index)

#normalize the dataset 
normal_train = normal_train_data(train)
normal_test = normal_test_data(train,test)


#get the x features and y 
x_ya_train = np.array(normal_train.drop(['Y'], axis=1))
y_ya_train = np.array(normal_train[['Y']])

x_ya_test = np.array(normal_test.drop(['Y'], axis=1))
y_ya_test = np.array(normal_test[['Y']])


#dataset 3

# concrete data
concrete = pd.read_csv('concreteData.csv')

#Split the data and get the training and testing data 
train=concrete.sample(frac=0.8,random_state=200) 
test=concrete.drop(train.index)

#normalize the dataset 
normal_train = normal_train_data(train)
normal_test = normal_test_data(train,test)

#get the x features and y 
x_con_train = np.array(normal_train.drop(['Y'], axis=1))
y_con_train = np.array(normal_train[['Y']])

x_con_test = np.array(normal_test.drop(['Y'], axis=1))
y_con_test = np.array(normal_test[['Y']])


# implement the GD

# cost function
def rmse_cost_func(y_hat,y):
    return np.sqrt(np.mean((y_hat-y)**2))

def gradient_of_cost(x,y,w):
    preds = predict(x,w)
    error = preds-y
    error_term = (1/len(x))*np.matmul(x.T,error)
    return error_term

def predict(x,w):
    return np.matmul(x,w)

# find a target linear model using gradient descent

def find_linear_regression_model(x,y):
    max_epochs = 50000
    alpha = 0.01 # learning rate
    tolerance = 0.01
    w= np.zeros(x.shape[1]).reshape(-1,1)
    loss_f = []
    n_epochs = 0
    
    while (n_epochs < max_epochs):
        
            n_epochs = n_epochs +1
            y_hat = predict(x,w)
            loss = rmse_cost_func(y_hat,y)
            loss_f.append(loss)
            diff = 1
        
            if (n_epochs > 2):
                prev = n_epochs -2
                curr = n_epochs -1
                diff = (loss_f[prev]-loss_f[curr])
            
            if (abs(diff) < tolerance):
                print ('converged')
                break
        
            else:
                grad = gradient_of_cost(x,y,w)
                next_w = w - alpha*grad
                w = next_w
            
    print ("Epoch: %d" %(n_epochs))        
    return w


#train the model of dataset 1: housing
w = find_linear_regression_model(x_house_train, y_house_train)

# prediction
y_hat_train = predict(x_house_train,w)
#cost of training
train_rmse = rmse_cost_func(y_hat_train,y_house_train)

# prediction
y_hat_test = predict(x_house_test,w)
#cost of test
test_rmse = rmse_cost_func(y_hat_test,y_house_test)

print('train_rmse: {0:.6f}'.format(train_rmse))
print('test_rmse: {0:.6f}'.format(test_rmse))


#train the model of dataset 2: yacht
w = find_linear_regression_model(x_ya_train, y_ya_train)

# prediction
y_hat_train = predict(x_ya_train,w)
#cost of training
train_rmse = rmse_cost_func(y_hat_train,y_ya_train)

# prediction
y_hat_test = predict(x_ya_test,w)
#cost of test
test_rmse = rmse_cost_func(y_hat_test,y_ya_test)

print('train_rmse: {0:.6f}'.format(train_rmse))
print('test_rmse: {0:.6f}'.format(test_rmse))



#train the model of dataset 3: concrete
w = find_linear_regression_model(x_con_train, y_con_train)

# prediction
y_hat_train = predict(x_con_train,w)
#cost of training
train_rmse = rmse_cost_func(y_hat_train,y_con_train)

# prediction
y_hat_test = predict(x_con_test,w)
#cost of test
test_rmse = rmse_cost_func(y_hat_test,y_con_test)

print('train_rmse: {0:.6f}'.format(train_rmse))
print('test_rmse: {0:.6f}'.format(test_rmse))


#dataset 1
normal_house = normal_train_data(house)
x_house = np.array(normal_house.drop(['Y'], axis=1))
y_house = np.array(normal_house[['Y']])



#dataset 2
normal_yacht = normal_train_data(yacht)
x_ya = np.array(normal_yacht.drop(['Y'], axis=1))
y_ya = np.array(normal_yacht[['Y']])



#dataset 3
normal_concrete = normal_train_data(concrete)
x_con = np.array(normal_concrete.drop(['Y'], axis=1))
y_con = np.array(normal_concrete[['Y']])


# find a target model using least square lost function ( normal equation)
# cost function
def rmse_cost_func(y_hat,y):
    return np.sqrt(np.mean((y_hat-y)**2))

def predict(x,w):
    return np.matmul(x,w)

def NE_linear_regression_model(x,y):
    first = np.linalg.inv(np.dot(x.T, x))
    second = np.dot(x.T, y)
    w = np.dot(first,second)
    
    return w
    


#dataset 1
#train the model 
w = NE_linear_regression_model(x_house, y_house)

# prediction
y_hat = predict(x_house,w)
#cost
rmse = rmse_cost_func(y_hat,y_house)

print('NormalEquation_rmse: {0:.6f}'.format(rmse))


#dataset 2
#train the model 
w = NE_linear_regression_model(x_ya, y_ya)

# prediction
y_hat = predict(x_ya,w)
#cost
rmse = rmse_cost_func(y_hat,y_ya)

print('NormalEquation_rmse: {0:.6f}'.format(rmse))


#dataset 3
#train the model 
w = NE_linear_regression_model(x_con, y_con)

# prediction
y_hat = predict(x_con,w)
#cost
rmse = rmse_cost_func(y_hat,y_con)

print('NormalEquation_rmse: {0:.6f}'.format(rmse))


