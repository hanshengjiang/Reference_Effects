#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:54:59 2020

@author: hanshengjiang
"""
'''
contain 
'''
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets
from IPython.display import display
import scipy.linalg
from itertools import combinations
from numpy import linalg
from numpy.linalg import matrix_rank
from scipy.sparse.linalg import svds, eigs
import time
from scipy.optimize import Bounds
from scipy.optimize import minimize
import pandas as pd
from scipy.stats import multivariate_normal
from scipy import integrate
from sklearn import linear_model
from datetime import datetime
import multiprocessing 
from sklearn import svm
from itertools import repeat

def generate_choice_prob(V,B,vB,alpha):
    '''
    Input: 
    V (n,d) d-dimensional feature vectors of n items, including no-purchase item
    n = 2
    B (d,*) each column of P is a d-dimension coefficient
    vB (d,*) each column of P is a d-dimension coefficient 
    alpha (*,1) 
    (theta and vartheta)
    => alpha and B together specify a distirbution of the parameters 
    
    Output:
    prob (n,) choice probabilitise of n items
                 for coefficient with distribution (alpha, P)
    
    '''
    
    n = len(V)
    (d,K) = np.shape(B)
    
    prob = np.zeros(n)
    
    eps = 1e-3
     
    for k in range(K):
        if np.dot(V[1],vB[:,k]) > eps:
            prob += np.array([0,alpha[k]])
        elif np.dot(V[1],vB[:,k]) < -eps:
            prob += np.array([alpha[k],0])
        else:
            prob += alpha[k]*np.exp(np.dot(V,B[:,k]) -\
                         np.logaddexp(np.dot(V[0],B[:,k]), np.dot(V[1],B[:,k])))
    return prob

def theta_lmo(b,vb,V_alltime,choice_ct,f):
    '''
    
    lmo: Linear Minimization Oracle
    
    Input: 
    
    b (d,)
    vb (d,)
    V_alltime (T,n,d) d-dimensional feature vectors of n items, including no-purchase item, of T time periods
    choice_ct (T,n) count of ppl choose each item at each time period
    f (T,n) current mixture likelihood vector
    
    Output:
    
    obj_func scalar, objective function of lmo on b
    
    '''
    (T,n,d) = np.shape(V_alltime)
    obj = np.zeros(T)
    prob_alltime = np.zeros((T,n))
    
    eps = 1e-3
    # count = 0
    for t in range(T):
        prob_alltime[t] = generate_choice_prob(V_alltime[t], \
                np.reshape(b,(d,1)), np.reshape(vb,(d,1)), np.array([1]))
        # if prob_alltime[t,0] > 0 and prob_alltime[t,0] < 1 :
        if np.dot(V_alltime[t,1,:],vb) > -eps and np.dot(V_alltime[t,1,:],vb) < eps:
            # count = count + 1
            obj[t] = -(2*choice_ct[t,1]-1)*prob_alltime[t,1]/np.dot(choice_ct[t,:],f[t,:]) 
    # print("theta_lmo theta", count)
    obj_func = np.sum(obj)/T
    return obj_func  


#-------------------
def altmax_sollmo(V_alltime,choice_ct,f,alpha, limiting_points = True):
    
    '''
    sollmo: solve linear minimization oracle
    
    Input: 

    V_alltime (T,n,d) d-dimensional feature vectors of n items, including no-purchase item, of T time periods
    choice_ct (T,n)
    f (T,n) current mixture likelihood vector
    
    Output:
    
    b_sol that minimizes lmo
    g corresponding atomic likelihood vector
    
    '''
    (T,n,d) = np.shape(V_alltime)
    l_bd_abs = 1000000
    u_bd_abs = 1000000
    bd_param = Bounds(l_bd_abs*np.array([-1,-1,-1,-1]),u_bd_abs*np.array([1,1,1,1]))
    
    # initialization 
    b_sol = np.zeros((d,1))
    vb_sol = np.zeros((d,1))
    

    iters_altmax = 10
    
    subprob_obj_rec = []
    
    for num in range(iters_altmax):
        
        #--------------------------------
        # theta step
        #--------------------------------
        # multiple initializations
        num_randomin = 10
        fun_sol = np.full((num_randomin,), np.inf)
        b_sol_array = np.zeros((d,num_randomin))
        vb_sol = np.zeros((d,1))
    
        for i in range(num_randomin):
            # initialize b0
            b0 = np.random.normal(0,10,(d,))
            
            OptResult = minimize(theta_lmo, b0, args = (vb_sol,V_alltime,choice_ct,f),\
                                 method = 'Powell', bounds= bd_param, \
                                 options={'disp':False, 'xtol':0.01, 'ftol':0.001})

            if OptResult.success == True:
                # print(b_sol_temp)
                b_sol_array[:d,i] = OptResult.x.ravel()
                fun_sol[i] = OptResult.fun
  
        
        s = np.argmin(fun_sol.ravel())
        b_sol = b_sol_array[:,s]
        # print("b_sol", b_sol)
        # print("fun_sol", fun_sol[s])
        #--------------------------------
        
        subprob_obj_new = 0
        for t in range(T):
            subprob_obj_new += (2*choice_ct[t,1]-1)/np.dot(choice_ct[t,:],f[t,:]) \
                * float(np.dot(V_alltime[t,1,:], vb_sol)>0)
            subprob_obj_new += (2*choice_ct[t,1]-1)/np.dot(choice_ct[t,:],f[t,:]) \
                * float(np.dot(V_alltime[t,1,:], vb_sol)==0) * \
                np.exp(np.dot(V_alltime[t,:,:],b_sol) - np.logaddexp(np.dot(V_alltime[t,0,:],b_sol),\
                              np.dot(V_alltime[t,1,:],b_sol)))[1]

        if len(subprob_obj_rec)==0 or subprob_obj_new > subprob_obj_rec[-1]:
            subprob_obj_rec.append(subprob_obj_new)
            
        #--------------------------------  
        
        #--------------------------------
        # vartheta step
        #--------------------------------
        
        if limiting_points == True:
            
            vb_sol_new = np.zeros((d,1))
            if np.sum(choice_ct[:,1]) > 0 and np.sum(choice_ct[:,0]) > 0:
                clf = svm.SVC(kernel='linear')
                coef_ = np.zeros(T)
                for t in range(T):
                    coef_[t] = 1/np.dot(choice_ct[t,:],f[t,:])*\
                                    np.exp(np.dot(V_alltime[t,:,:],b_sol) \
                                    -np.logaddexp(np.dot(V_alltime[t,0,:],b_sol),\
                                                  np.dot(V_alltime[t,1,:],b_sol)))[int(choice_ct[t,1])]
                                    
                eps = np.finfo(float).tiny # add tiny value to make sure sample weights are not identically 0
                clf.fit(V_alltime[:,1,1:],choice_ct[:,1],sample_weight= coef_+eps)
                vb_sol_new[0] = clf.intercept_[0]
                vb_sol_new[1:] = np.reshape(clf.coef_[0],(d-1,1))
                
                
                # update
        #        subprob_obj = 0
        #        for t in range(T):
        #            subprob_obj += (2*choice_ct[t,1]-1)/np.dot(choice_ct[t,:],f[t,:]) \
        #                * float(np.dot(V_alltime[t,1,:], vb_sol)>0)
        #            subprob_obj += (2*choice_ct[t,1]-1)/np.dot(choice_ct[t,:],f[t,:]) \
        #                * float(np.dot(V_alltime[t,1,:], vb_sol)==0) * \
        #                np.exp(np.dot(V_alltime[t,:,:],b_sol) - np.logaddexp(np.dot(V_alltime[t,0,:],b_sol),np.dot(V_alltime[t,1,:],b_sol)))[1]
        #
        #        
                subprob_obj_new = 0
                for t in range(T):
                    subprob_obj_new += (2*choice_ct[t,1]-1)/np.dot(choice_ct[t,:],f[t,:]) \
                        * float(np.dot(V_alltime[t,1,:], vb_sol_new)>0)
                    subprob_obj_new += (2*choice_ct[t,1]-1)/np.dot(choice_ct[t,:],f[t,:]) \
                        * float(np.dot(V_alltime[t,1,:], vb_sol_new)==0) * \
                        np.exp(np.dot(V_alltime[t,:,:],b_sol) - np.logaddexp(np.dot(V_alltime[t,0,:],b_sol),\
                                      np.dot(V_alltime[t,1,:],b_sol)))[1]
        
                if len(subprob_obj_rec)==0 or subprob_obj_new > subprob_obj_rec[-1]:
                    subprob_obj_rec.append(subprob_obj_new)
                    # print("vb_sol", vb_sol_new)
                    # print(vb_sol_new,subprob_obj_new)
                    vb_sol = np.copy(vb_sol_new)
                    
        #--------------------------------
        
    #-------------------------------------
    #generate likelihood vector from b_sol and vb_sol
    g = np.zeros((T,n,1))
    for t in range(T):
        g[t,:,0] = generate_choice_prob(V_alltime[t,:,:],np.reshape(b_sol,(d,1)),\
         np.reshape(vb_sol,(d,1)),np.array([1]))
    return b_sol, vb_sol, g


#-------------------

def reoptimization(f,P,f_new,choice_ct,alpha):
    '''
    # reoptimization
    
    Input:
    f (T,n) current  
    P (T,n,*) each column is a T*n likelihood vector
    f_new (T,n) new atomic likelihood vector 
    alpha (*,) current weights
    
    Ouput:
    return new f, and f as a convex combination of columns of P with coefficients alpha
    
    '''
    iter = 5000
    
    (T,n,K) = np.shape(P)
    
    f_new = np.reshape(f_new,(T,n))
    f = np.reshape(f,(T,n))
    
    #append new vector g to the end of actice set P
    P = np.append(P,np.reshape(f_new,(T,n,1)),axis = 2)
    
    #append 0 to the end  of alpha, to correspond to the new g
    alpha = np.append(alpha,0)
    k = len(alpha)
    
    for t in range(1,iter):
        g = 1/f
        
        lin_prob = np.zeros(K)
        #choose s by solving lmo
        for i in range(K):
            lin_prob[i] = np.sum(np.multiply(choice_ct,np.multiply(P[:,:,i],np.reshape(g,(T,n)))))
        s = np.argmax(lin_prob) 
        
        
        #gamma step-size
        gamma = 2/(t+3)
        
        f = (1-gamma)*f +gamma*P[:,:,s]
        temp_alpha = np.zeros(k)
        temp_alpha[s] = 1
        alpha = (1-gamma)*alpha+gamma*temp_alpha
        

    return f,P,alpha

def CGM(V_alltime,choice_ct,iters, r_seed = 626):
    '''
    Input:
    V_alltime (T,n,2d) d-dimensional feature vectors of n items, in T time periods
    choice_ct (T,n) integer, count of purchases for n items, in T time periods 
    iters scalar number of large iterations in CGM
    
    Output: 
    f, 
    B (2d, T),
    alpha, 
    L_rec,
    temp
    '''
    
    np.random.seed(r_seed)

    (T,n,d) = np.shape(V_alltime)
    # n = 2 in single-product case
    
    #record L, b_sol
    L_rec = []
    
    # initialize b0 and f
    b0 = np.random.uniform(-1,1,(d,1))
    # heuristic sign correction
    b0[0] = np.minimum(b0[0],-b0[0])
    b0[1] = np.maximum(b0[1],-b0[1])
    b0[2] = np.maximum(b0[2],-b0[2])
    b0[3] = np.maximum(b0[3],-b0[3])
        
   
    
    # initialize B : active set of coefficients
    B = np.zeros((d,1))
    B[:,0] = b0.ravel()
    vB = np.zeros((d,1))
    
    # intialize alpha
    alpha = np.array([1]) 
    
    # iniitialize f
    f = np.zeros((T,n))
    for t in range(T):
        f[t,:] = generate_choice_prob(V_alltime[t,:,:],B,vB,alpha)
        
    # initialize P: active set of likelihood vectors
    P = np.zeros((T,n,1))
    P[:,:,0] = np.reshape(f,np.shape(P[:,:,0]))
    
    iter_ct = 0
    temp = np.inf
    ct = 0 # number of small likelihood change
    while iter_ct < iters:
        iter_ct = iter_ct + 1 
        
        # solve LMO
        b_sol,vb_sol,g = altmax_sollmo(V_alltime,choice_ct,f,alpha)
        
        #if t%10 == 0:
        # print('iteration:', t)
        # print('b_sol',b_sol)
               
        g = np.reshape(g,(T,n))
        
        # fully corrective step wrt current active set P and new likelihood vecrtor g
        f,P,alpha = reoptimization(f,P,g,choice_ct,alpha)
        alpha = np.array(alpha)
        
       
        # update B, vB
        B = np.append(B,np.reshape(b_sol,(d,1)),axis = 1)
        vB = np.append(vB,np.reshape(vb_sol,(d,1)),axis = 1)
        
        # record the change of neg-log likelihood function
        temp_n = np.sum(np.multiply(choice_ct,np.log(1/f))) 
        if temp - temp_n < 1e-6:
            ct += 1
        if ct > iters:
            break
        temp = temp_n
        L_rec.append(temp)
        print('negative log likelihood',temp)
        
    #------------------------------#
    # pruning, remove zero weights
    eps = 1e-3
    mask = np.argwhere(alpha>eps).ravel()
    # P = P[:,:,mask]
    alpha = (alpha[mask]).ravel()
    B = B[:,mask]
    vB = vB[:,mask]
    
    order = np.argsort(-alpha)
    alpha = alpha[order]
    B = B[:, order]
    vB = vB[:, order]
    
    for t in range(T):
        f[t,:] = generate_choice_prob(V_alltime[t,:,:],B,vB,alpha)
    #------------------------------#
    
    return f, B, vB,alpha, L_rec


#------------------------------------------------------
# auxilliary functions
#------------------------------------------------------
def predict_probability(B,vB,alpha, V_test):
    '''
    Input:
    V_test (T,n,d) d-dimensional feature vectors of n items, in T time periods
    B (d,k)    components from estimation
    alpha (k,) weights from estimation
    
    Output:
    proba choice probabilities
    '''
    (T,n,d) = V_test.shape
    (d,k) = B.shape
    
    proba = np.zeros((T,n))
    for t in range(T):
        V = V_test[t,:,:]
        proba[t] = generate_choice_prob(V,B, vB,alpha)
    return proba
    
def linear_model(V_lin, D):
    '''
    Input 
    V_lin
    each row looks like (1, p, (r-p)_+, (r-p)_-)
    
    Outpu
    coefficents
    training error (based on RMSE)
    '''
    
    # fit_intercept = False -> no extra constant intercept
    reg = linear_model.LinearRegression(fit_intercept = False).fit(V_lin,D)
    
    error = np.linalg.norm(D - reg.predict(V_lin))/np.sqrt(len(D))
    
    return reg.coef_, error

def sub_sample(V_alltime, choice_ct, sub_sampling):
    # get subsample of data
    
    
    # keep the ratio of purchase vs. no-purchse unchanged
    purchase_index = np.argwhere(choice_ct[:,0] == 0).ravel()
    no_purchase_index = np.argwhere(choice_ct[:,0] == 1).ravel()
    
    ratio_p = len(purchase_index)/len(choice_ct)
    num_p = int(sub_sampling * ratio_p)+1
    num_np = sub_sampling - num_p
    
    sample_purchase_index = np.array(np.random.choice(list(purchase_index), num_p, replace = False) )
    sample_no_purchase_index = np.array( np.random.choice(list(no_purchase_index), num_np, replace = False) )
    
    # combine to get sample index
    sample_index = np.hstack((sample_purchase_index, sample_no_purchase_index))
    
    perm  = np.random.permutation(num_p + num_np)
    sample_index = sample_index[perm]
    
    V_alltime = np.copy(V_alltime[sample_index,:,:])
    choice_ct = np.copy(choice_ct[sample_index,:])
    
    return V_alltime, choice_ct

def sub_sample_daily(V_alltime, choice_ct, V_daily, sub_sampling):
    # get subsample of data
    
    
    # keep the ratio of purchase vs. no-purchse unchanged
    purchase_index = np.argwhere(choice_ct[:,0] == 0).ravel()
    no_purchase_index = np.argwhere(choice_ct[:,0] == 1).ravel()
    
    ratio_p = len(purchase_index)/len(choice_ct)
    num_p = int(sub_sampling * ratio_p)+1
    num_np = sub_sampling - num_p
    
    sample_purchase_index = np.array(np.random.choice(list(purchase_index), num_p, replace = False) )
    sample_no_purchase_index = np.array( np.random.choice(list(no_purchase_index), num_np, replace = False) )
    
    # combine to get sample index
    sample_index = np.hstack((sample_purchase_index, sample_no_purchase_index))
    
    perm  = np.random.permutation(num_p + num_np)
    sample_index = sample_index[perm]
    
    V_alltime = np.copy(V_alltime[sample_index,:,:])
    choice_ct = np.copy(choice_ct[sample_index,:])
    V_daily = np.copy(V_daily[sample_index,:])
    
    return V_alltime, choice_ct, V_daily

def neg_log_likelihood(B, alpha, V_alltime, choice_ct):
    '''
    calculate the likelihood based on 
    parameter B, alpha
    and testing data V and choice_ct
    '''
    (T,n,d) = V_alltime.shape
    f = np.zeros((T,n))
    for t in range(T):
        f[t,:] = generate_choice_prob(V_alltime[t,:,:],B,alpha)
    
    neg_log_likelihood = 0.0
    for t in range(T):
        neg_log_likelihood += -np.log( np.sum(np.multiply(choice_ct[t,:], f[t,:])) )
    return neg_log_likelihood

