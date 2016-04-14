
# coding: utf-8

# In[7]:

import numpy as np
import math as math

'''
This function is to generate data with binary label, based on the model of Logistic Regression
'X' is sampled from a uniform distribution between [0,1]
hypothesis 'h' is generated from a integer uniform distribution between [-5,5]
the noise level 'sigma' is generated from a normal distribution with mean 0 and std sigma
label y is generated in the way below:
first generate y_linear=X*w+sigma as a linear problem
and then use logistic regression model to transform y_linear to a probability between [0,1]
and then generate a variable 'e' uniformly from [0,1], 
if y_linear>e ,set the label to be 1, otherwise set the label to be 0.

Input: number of samples N,dimension of data d, noise level sigma

Output: generate X, label y, realy hypothesis h
'''

def generate_data_lr(N,d,sigma):
    # generate X uniform distributed between[-1,1]
    X=np.random.uniform(-1,1,size=(N,d))
    # generate noise normal distributed, mean 0, variance sigam^2
    noise=np.random.normal(0.0,sigma,(N,1))
    # add a collumnn of b so that the model is y=X*beta+b, 
    b=np.ones((N,1))
    X_modify=np.concatenate((X,b),axis=1)
    # generate hypothesis of the linear classification, it is d+1 dimension vector
    h=np.random.uniform(-5.0,5.0,size=(d+1,1))
    # generate value of y if it is a linear regression problem
    y_linear=np.dot(X_modify,h)+noise
    # transform y_linear into a  0-1 label based on logistic regression model
    y=1 / (1 + np.exp(-y_linear))
    for i in range(0,N):
        e=np.random.uniform(0,1,size=(1,1))
        if y[i,0]>=e:
            y[i,0]=1
        else:
            y[i,0]=0
    
    return X,y,h





'''
function generate_cost_SampleGroup is to generate cost correlated with their group
first randomly assign a group label for each sample, 
In each group, the cost is a normal distribution with a specific mean and variance
Input:
'Group_Number' is a integer showing the number of groups
'Cost_Mean' is a 1-D vector contains the cost mean of each group
'Cost_Var' is a 1-D vector contains the cost variance of each group
'X' generated data
'cost_sg' is a sample
'''


def generate_cost_SampleGroup(Group_Number,Cost_Mean,Cost_Var,X):
    # number of samples
    N=X.shape[0]
    # initialize the cost vector of data
    cost_sg=np.zeros((N,1))
    # randomly assign group label for each sample
    group_label=np.random.randint(1,Group_Number,(N,1))
    # sample the cost for each sample
    for i in range(0,N):
        mean=Cost_Mean[int(group_label[i,0]),0];
        var=Cost_Var[int(group_label[i,0]),0];
        cost=np.random.normal(mean,var,1);
        cost_sg[i,0]=cost;
    
    cost_sg=cost_sg/np.sum(cost_sg, axis=0)*N
    return cost_sg

'''
function convex_cff and linear_cff are two functions that map a specific feature to a cost
Input 'a','b','c' the parameter in the function ,'x' the feature
'''

def convex_cff(a,b,c,x):
    cost=a*np.multiply(x,x)+b*x+c
    return cost

def linear_cff(a,b,x):
    cost=a*x+b
    return cost

'''
This function is for generate features correlated with features, 
Input: X, generated features
       coeff: cost function parameter for every feature,
       for exaple, if I use convex_cff, then for each feature i I have [coeff[i,0],coeff[i,1],coeff[i,2]]
       corresponding to a,b c in the function
       w: weight parameter for every features
Output: the normalized output cost correlated with features
'''


def generate_cost_Features(X,coeff,w):
    N=X.shape[0]
    d=X.shape[1]
    # calculate the cost for each features
    for i in range(0,d):
        X[:,i]=convex_cff(coeff[i,0],coeff[i,1],coeff[i,2],X[:,i])
    cost_fts=np.dot(X,w)
    # normalized cost, for each data set the total cost is the number of its data points
    cost_fts=cost_fts/np.sum(cost_fts, axis=0)*N
    return cost_fts


'''
main part of the data generating file
'''

#Initialize the number of samples, data points and noise level
N=10000
d=5
sigma=0.01
# generate data based on logistic regression model
[X_lr,y_lr,h_lr]=generate_data_lr(N,d,sigma)
print X_lr,y_lr
np.savetxt('X_lr.txt', X_lr, delimiter=',')
np.savetxt('y_lr.txt', y_lr, delimiter=',')
np.savetxt('h_lr.txt',h_lr,delimiter=',')

#Initialize the parameter for generate the cost correlated with sample groups
Group_Number=5;
Cost_Mean=np.random.randint(5,10,(Group_Number,1))
sigma_cost=0.01
Cost_Var=sigma_cost*np.random.randint(1,5,(Group_Number,1))
cost_sg=generate_cost_SampleGroup(Group_Number, Cost_Mean,Cost_Var,X_lr)
print cost_sg   
np.savetxt('cost_sg.txt', cost_sg, delimiter=',')

#Initialize the parameter for generate the cost correlated with features
Group_Number=5;
coeff=np.random.uniform(0.0,1.0,(d,3))
w=np.random.uniform(0.0,1.0,(d,1))
cost_fts=generate_cost_Features(X_lr,coeff,w)
print cost_fts
np.savetxt('cost_fts.txt', cost_sg, delimiter=',')


# In[ ]:




# In[ ]:



