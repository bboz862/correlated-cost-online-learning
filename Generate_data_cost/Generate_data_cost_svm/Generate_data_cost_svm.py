
# coding: utf-8

# In[12]:

import numpy as np
import math as math

'''
This function is to generate data with binary label, based on the model of support vector machine,
Which is just to find a separating hyperplane of the data
'X' is sampled from a uniform distribution between [0,1]
hypothesis 'h' is generated from a integer uniform distribution between [-5,5]
the noise level 'sigma' is generated from a normal distribution with mean 0 and std sigma
label y is generated in the way below:
if Ax+b>0 then y=1 else y=0
reference A and b are stored in the parameter 'hyp'

Input: number of samples N,dimension of data d, noise level sigma

Output: generate X, label y, realy hypothesis hyp
'''

def generate_data_svm(N,d,sigma):
    # generate X uniform distributed between[-1,1]
    X=np.random.uniform(-1,1,size=(N,d))
    # generate noise normal distributed, mean 0, variance sigam^2
    noise=np.random.normal(0.0,sigma,(N,1))
    # generate hypothesis of the linear classification, it is d+1 dimension vector
    h=np.random.uniform(-5.0,5.0,size=(d,1))
    print h
    # generate value of y if it is a linear regression problem
    y_linear=np.dot(X,h)+noise
    print y_linear
    avg_y=np.sum(y_linear,axis=0)/N
    print avg_y
    b=avg_y*np.ones((N,1))
    # transform y_linear into a  0-1 label based on logistic regression model
    y=y_linear-b
    #y=1 / (1 + np.exp(-y_linear))
    for i in range(0,N):
        if y[i,0]>=0:
            y[i,0]=1
        else:
            y[i,0]=0
    hyp=np.zeros((d+1,1))
    hyp[0:d,0]=h[0:d,0]
    hyp[d,0]=-avg_y
    return X,y,hyp




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
# generate data based on separating hyperplane model
[X_svm,y_svm,h_svm]=generate_data_svm(N,d,sigma)
print X_svm,y_svm
np.savetxt('X_svm.txt', X_svm, delimiter=',')
np.savetxt('y_svm.txt', y_svm, delimiter=',')
np.savetxt('h_svm.txt',h_svm,delimiter=',')

#Initialize the parameter for generate the cost correlated with sample groups
Group_Number=5;
Cost_Mean=np.random.randint(5,10,(Group_Number,1))
sigma_cost=0.01
Cost_Var=sigma_cost*np.random.randint(1,5,(Group_Number,1))
cost_sg=generate_cost_SampleGroup(Group_Number, Cost_Mean,Cost_Var,X_svm)
print cost_sg   
np.savetxt('cost_sg.txt', cost_sg, delimiter=',')

#Initialize the parameter for generate the cost correlated with features
Group_Number=5;
coeff=np.random.uniform(0.0,1.0,(d,3))
w=np.random.uniform(0.0,1.0,(d,1))
cost_fts=generate_cost_Features(X_svm,coeff,w)
print cost_fts
np.savetxt('cost_fts.txt', cost_sg, delimiter=',')


# In[ ]:



