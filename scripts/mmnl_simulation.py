#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:39:49 2020

@author: hanshengjiang
"""

from py_estimation import *

def simulated_uniform_data(T,n,d,num_ppl,params):
    '''
    Input:
    T
    n : number of items
    d : number of features
    num_ppl: number of ppl at each time period
    b_dist (d,*)
    w (*,) weights of different coefficients
    params:  ground truth
    
    Output:
    V_alltime
    choice_ct
    prob_alltime: choice proabbility conditioning on thetavec
    D_alltime: expected demand from a random consumer
    '''
 
    (mu1,sigma1,mu2,sigma2,BH) = params
    
    V_alltime = np.ones((T,n,d))
    
    # make features be like (1 -p, (r-p)_+, (r-p)_-)
    V_alltime[:,1,1] = - np.abs(np.random.normal(mu1,sigma1,T))
    
    rminusp = np.random.normal(mu2,sigma2,T)
    
    V_alltime[:,1,2] = np.maximum(rminusp, np.zeros(T))
    V_alltime[:,1,3] = np.minimum(rminusp, np.zeros(T))
     
    
    #no purchase item
    V_alltime[:,0,:] = np.zeros((T,d))
    
    
    choice_ct = np.zeros((T,n))
    
    prob_alltime = np.zeros((T,2))
    
    D_alltime = np.zeros((T,2))
    
    #BH = 10 # responsiveness parameter uniform distributed over [0,BH]^4
    for t in range(T):
        
        x1 = V_alltime[t,1,1]
        x2 = V_alltime[t,1,2]
        x3 = V_alltime[t,1,3]
        #expected demand given current features
        #D_uniform_fast provides approximation
        D_alltime[t,1] = D_uniform_fast(x1,x2,x3,(BH/2, BH/2, BH/2, BH/2),BH)
        D_alltime[t,0] = 1 - D_alltime[t,1]
        
        prob = np.zeros(n,)
        for id_ppl in range(num_ppl):
            
            # sample consumer responsiveness parameters
            b = np.random.uniform(0,BH,d)
            
            # b[0] = b[0] - BH*0.5
            
            # sign correction
            # b[1] = -b[1]
            
            vb = np.zeros(d)
            
            #choice probabilities given b
            prob = generate_choice_prob(V_alltime[t,:,:],np.reshape(b,(d,1)),np.reshape(vb,(d,1)),np.array([1]))
            prob_alltime[t] = prob
            
            #generate choice according the choice probabilities
            choice = np.random.choice(n,1,p = prob.ravel())
            
            #record choice count
            choice_ct[t,choice]+=1.0
           
    return V_alltime,choice_ct,prob_alltime, D_alltime

def D_single(a,b,c_pos,c_neg,x1,x2,x3):
    '''
    r: reference price
    p: price
    
    a, b(<=0), c_pos, c_neg: coefficients
    '''
    u = a + b * x1 + c_pos * x2 + c_neg * x3

    demand = np.exp(u)/(1+np.exp(u))
    return demand
    
def D_uniform(x1,x2,x3,coefficients,BH):
    '''
    one period demand, logistic demand function
    mixing distribution is a uniform distribution
    
    Input
    --------
    coefficients: integratation range or parameter distribution range
    p: price
    r: reference price
    '''
    #BH = 10
    (demand_uniform, abserr) = integrate.nquad(D_single, coefficients, args = (x1,x2,x3))
    M = BH**4
    demand_uniform = demand_uniform/M
    return demand_uniform

def D_uniform_fast(x1,x2,x3,coefficients,BH):
    '''
    one period demand, logistic demand function
    mixing distribution is a uniform distribution
    
    Input
    --------
    coefficients: integratation range or parameter distribution range
    p: price
    r: reference price
    '''
    num_sample = 500
    demand_uniform = 0
    
    c = np.array(coefficients)
    
    for i in range(num_sample):
        b = np.random.uniform(-BH/2,BH/2,4)
        b = b + c
        
        u = b[0] + b[1] * x1 + b[2] * x2 + b[3] * x3
        demand_uniform += np.exp(u)/(1+np.exp(u))
    demand_uniform = demand_uniform/num_sample
    return demand_uniform



def simulated_Gaussian_data(T,n,d,num_ppl):
    '''
    Input:
    T
    n : number of items
    num_ppl: number of ppl at each time period
    b_dist (d,*)
    w (*,) weights of different coefficients
    
    Output:
    V_alltime
    choice_ct
    
    '''
    
    w1 = 0.4
    w2 = 0.6
    mean1 = (3,-1,3,-1)
    cov1 = [[0.2, -0.1, 0,0], [-0.1, 0.4,0,0], [0,0,0.2,-0.1],[0,0,-0.1,0.4]]
    mean2 = (-1,1,-1,1)
    cov2 = [[0.3,0.1,0,0],[0.1,0.3,0,0],[0,0,0.3,0.1],[0,0,-0.1,0.4]]
    
    
    #mu = 0, sigma = 1.5
    V_alltime = np.random.normal(0,1.5,(T,n,d))
    V_alltime[:,:,0] = np.ones((T,n))
    V_alltime[:,1,1] = - V_alltime[:,1,1]
    
    #no purchase item
    V_alltime[:,0,:] = np.zeros((T,d))
    
    
    choice_ct = np.zeros((T,n))
    
    for t in range(T):
        prob = np.zeros(n,)
        for id_ppl in range(num_ppl):
            
            #sample b from mixture distributions
            z = np.random.uniform(0,1)
            if z <= w1:
                b = np.random.multivariate_normal(mean1,cov1)
            else:
                b = np.random.multivariate_normal(mean2,cov2)
            
            #choice probabilities given b
            prob = generate_choice_prob(V_alltime[t,:,:],np.reshape(b,(d,1)),np.array([1]))
            
            #generate choice according the choice probabilities
            choice = np.random.choice(n,1,p = prob.ravel())
            
            #record choice count
            choice_ct[t,choice]+=1.0
            
    return V_alltime,choice_ct


def simulated_data(T,n,d,num_ppl,b_dist,w):
    '''
    Input:
    T
    n : number of items
    d
    num_ppl: number of ppl at each time period
    b_dist (d,K)
    w (K,) weights of different coefficients
    
    Output:
    V_alltime
    choice_ct
    
    '''
    V_alltime = np.random.uniform(3,5,(T,n,d))
    V_alltime[:,1,1] = - V_alltime[:,1,1]
    V_alltime[:,:,0] = np.ones((T,n))
    
    #no purchase item
    V_alltime[:,0,:] = np.zeros((T,d))
    
    
    choice_ct = np.zeros((T,n))
    K = len(w)
    prob = np.zeros((K,n))
    
    for t in range(T):
        for k in range(K):
            prob[k] = generate_choice_prob(V_alltime[t,:,:],np.reshape(b_dist[:,k],(d,1)),np.array([1]))
        for id_ppl in range(num_ppl):
            z = np.random.choice(K,1,p = w.ravel()) 
            choice = np.random.choice(n,1,p = prob[z].ravel())
            choice_ct[t,choice]+=1.0
    return V_alltime,choice_ct

def cee(p,q):
    '''
    total cross entropy error: \sum_{i=1}^n H(p_i,q_i)
    '''
    p = np.array(p).ravel()
    S = len(p)
    err = 0
    for i in range(S):
        err = err - p[i]*np.log(q[i]) - (1-p[i])*np.log(1-q[i])
    return err



