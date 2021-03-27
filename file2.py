#!/usr/bin/env python
# coding: utf-8

# In[63]:


#load the packages
import numpy as np
import pandas as pd


# In[64]:


#read the data
lab2DF = pd.read_csv('Lab2.csv')


# In[ ]:





# In[65]:


#define SSE score function

def SSE_1(predict,test_y):
    SSE = 0
    size = predict.size
    for i in range(0,size):
        SSE_part = (predict[i]- test_y[i])**2
        SSE = SSE + SSE_part
        
    return SSE


# In[66]:


# define liear model using normal equations 

def NE_linear_regression_model(x,y):
    first = np.linalg.inv(np.dot(x.T, x))
    second = np.dot(x.T, y)
    w = np.dot(first,second)  
    return w

def predict(x,w):
    return np.matmul(x,w)


# In[67]:


# define 1o-folds cv and get SSE scores

def CV_linear_1(lab2DF,n):
    
    lab2DF['x0'] = 1 #add the bias 
    shuffled = lab2DF.sample(frac=1)
    result = np.array_split(shuffled, n)   # ramdomly split the data 
    SSE_full = 0
    SSE_part = 0
    
    for i in range (0,n):
        test = result[i]
        dropindex = result[i].index
        train = lab2DF.drop(dropindex)
        train_x = train.iloc[:,train.columns != 'y']
        test_x = test.iloc[:,test.columns != 'y']
        
        #make the train and test data to fit the linear model
        train_y_a = np.array(train.iloc[:,train.columns == 'y'])
        train_x_a = np.array(train.iloc[:,train.columns != 'y'])
        
        test_y_a = np.array(test.iloc[:,test.columns == 'y'])
        test_x_a = np.array(test.iloc[:,test.columns != 'y'])
        
        #prediction
        w = NE_linear_regression_model(train_x_a, train_y_a)
        predict_1 = predict(test_x_a,w)
        SSE_i = SSE_1(predict_1,test_y_a)
        SSE_full = SSE_i + SSE_full
        
        # get the sub-data--2-predictors
        train_x_part = train_x.iloc[:, : 3]
        train_x_part['x0'] = 1    #add bias
        train_x_part = np.array(train_x_part) 
        
        test_x_part = test_x.iloc[:, : 3]
        test_x_part['x0'] = 1   #add bias
        test_x_part = np.array(test_x_part)  
        
        #prediction
        w = NE_linear_regression_model(train_x_part, train_y_a)
        predict_1 = predict(test_x_part,w)
        SSE_i = SSE_1(predict_1,test_y_a)
        SSE_part = SSE_i + SSE_part
        
    return SSE_full, SSE_part


# In[ ]:





# In[68]:


# repeat 20 times

def loop_CV_linear(lab2DF,n,m):
    for i in range (0,m):
        output = CV_linear_1(lab2DF,n)
        print (i,output)
    return(output)


# In[69]:


loop_CV_linear(lab2DF,10,20)


# In[ ]:




