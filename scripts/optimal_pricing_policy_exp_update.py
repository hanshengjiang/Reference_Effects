#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:36:49 2020

@author: hanshengjiang
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize
import time
# change plot fonts
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix",
     "font.size": 16}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

from scipy import integrate

def u(r,p,coefficients):
    '''
    utility function
    p: price
    r: reference price
    
    a,b,c_pos,c_neg all >=0
    
    '''
    (a ,b, c_pos, c_neg) = coefficients
    u = a - b * p + c_pos * np.maximum(r - p, 0) + c_neg * np.minimum(r - p, 0)
    return u


def R_single(a,b,c_pos,c_neg,r,p):
        '''
        r: reference price
        p: price
        
        a, b, c_pos, c_neg: coefficients
        '''
        u = a - b * p + c_pos * np.maximum(r - p, 0) + c_neg * np.minimum(r - p, 0)
        if u > 100:
            revenue = p
        elif u < -100:
            revenue = 0
        else:
            temp = 1/(1+np.exp(-u))
            revenue = p * temp
        return revenue
    
def R_uniform(r,p,coefficients):
    '''
    one period revenue, logistic demand function
    mixing distribution is a uniform distribution over [bL,bH]
    [bL,bH] integratation range or parameter distribution range
    p: price
    r: reference price
    '''
    # need to set the integral accuracy lower
    # --otherwise the intergal does not converge
    BH = np.array(coefficients[3])[1]
    (revenue_uniform, abserr) = integrate.nquad(R_single, coefficients,\
    args = (r,p), opts = [{'epsabs': 1e-6}, {'epsabs': 1e-6}, {'epsabs': 1e-6}, {'epsabs': 1e-6},])
    return revenue_uniform/(BH**4)

def R_uniform_fast(r,p,coefficients):
    '''
    one period revenue (approximated), logistic demand function
    mixing distribution is a uniform distribution over [bL,bH]
    [bL,bH] integratation range or parameter distribution range
    p: price
    r: reference price
    
    coefficients: ( coordiates of the center, edge length of cube)
    '''
    # need to set the integral accuracy lower
    # --otherwise the intergal does not converge
    c = np.array(coefficients[0])
    BH = float(coefficients[1])
    
    # print(BH)
    num_sample = 500
    revenue_uniform = 0
    for i in range(num_sample):
        b = np.random.uniform(-BH/2,BH/2,4)
        b = b + c
        
        u = b[0] - b[1] * p + b[2] * np.maximum(r - p, 0) + b[3] * np.minimum(r - p, 0)
        if u > 100:
            revenue_uniform += p
        elif u < -100:
            revenue_uniform += 0
        else:
            temp = 1/(1+np.exp(-u))
            revenue_uniform += p*temp
    revenue_uniform = revenue_uniform/num_sample
    # print(revenue_uniform)
    return revenue_uniform


def R(r,p,coefficients,weights):
    '''
    one period revenue, logistic demand function
    multi-segment consumers
    p: price
    r: reference price
    '''
    revenue = 0
    coefficients = np.array(coefficients)
    num_seg = int(len(coefficients)/4)
    for i in range(num_seg):
        # this number is set to be 26
        if u(r,p,coefficients[4*i:4*(i+1)]) > 100:
            # print('\n++++++++++++++++++++++++++')
            # print('overflow encountered in exp(+inf)')
            # print('++++++++++++++++++++++++++\n')
            revenue += p * weights[i]
        elif u(r,p,coefficients[4*i:4*(i+1)]) < -100:
            revenue += 0
        else:
            revenue += p*np.exp(u(r,p,coefficients[4*i:4*(i+1)]))/(1+ np.exp(u(r,p,coefficients[4*i:4*(i+1)])))*weights[i]
    return revenue

def R_ext(r,p,coefficients,weights):
    '''
    one period revenue, logistic demand function
    multi-segment consumers
    p: price
    r: reference price
    
    coefficients: includ both B and vB
    '''
    revenue = 0
    coefficients = np.array(coefficients)
    num_seg = int(len(coefficients)/8)
    
    v = (1, -p, max(r-p,0), min(r-p,0))
    
    eps = 1e-3
    
    for i in range(num_seg):
        if np.dot(v,coefficients[8*i+4:8*i+8]) > eps:
            revenue += p * weights[i]
        elif np.dot(v,coefficients[8*i+4:8*i+8]) < -eps:
            revenue += 0
        else:
            revenue += p*weights[i]*np.exp(np.dot(v,coefficients[8*i:8*i+4]) -\
                         np.logaddexp(0, np.dot(v,coefficients[8*i:8*i+4])))
    return revenue

def R_lin(r,p,coefficients,weights):
    '''
    one period revenue when the demand function is piece-wise linear
    multi-segment consumers
    
    ****lower bounded by zero****
    
    Input: 
    r reference price
    p current price
    coefficients 4*k coefficients for k segements in a sequential order
    
    Output:
    
    revenue from all segments
    '''
    revenue = 0
    coefficients = np.array(coefficients)
    
    #
    num_seg = int(len(coefficients)/4)
    for i in range(num_seg):
        revenue += p*max(u(r,p,coefficients[4*i:4*(i+1)]),0)*weights[i]
    return revenue

def D(r,p,coefficients,weights):
    '''
    one period demand, logistic demand function
    multi-segment consumers
    p: price
    r: reference price
    
    
    return
    demand
    '''
    demand = 0
    coefficients = np.array(coefficients)
    num_seg = int(len(coefficients)/4)
    
    
    for i in range(num_seg):
        if u(r,p,coefficients[4*i:4*(i+1)]) >9000:
            demand += 1 * weights[i]
        elif u(r,p,coefficients[4*i:4*(i+1)]) <- 9000:
            demand += 0
        else:
            demand += np.exp(u(r,p,coefficients[4*i:4*(i+1)]))/(1+ np.exp(u(r,p,coefficients[4*i:4*(i+1)])))*weights[i]
    return demand




def D_lin(r,p,coefficients,weights):
    '''
    one period demand when the demand function is piece-wise linear
    multi-segment consumers
    
    ****lower bounded by zero****
    
    Input: 
    r reference price
    p current price
    coefficients 4*k coefficients for k segements in a sequential order
    
    Output:
    
    demand from all segments
    '''
    demand = 0
    coefficients = np.array(coefficients)
    
    #
    num_seg = int(len(coefficients)/4)
    for i in range(num_seg):
        # heuristic fix
        demand += p* max(u(r,p,coefficients[4*i:4*(i+1)]),0)*weights[i]
    return demand

def non_decreasing(x):
    dx = np.diff(x)
    return np.all(dx >= 0)

def inf_hor_pricing_pricerange(L,H,theta,epsilon,T,gamma,coefficients,weights,func):
    '''
    Input:
    L: lower bound of price
    H: upper bound of price
    theta: memory parameter of prices
    epsilon: accuracy of price discretization
    T: number of time periods
    gamma: discounted factor
    coefficients:  u = a - b * p + c_pos * np.maximum(r - p, 0) + c_neg * np.maximum(p - r, 0) utility model
    
    Output:
    V: V[i] = revenue for infinite horizon when the first (reference) price is price_list[i]
    mu: mu[i] = optimal next reference price in the next time period given reference price is price_list[i]
    '''
    if T != np.inf:
        raise ValueError("Must be infinite horizon!")
    
    # decimals for rounding the value function
    decimals_ = 100
    
    price_list = np.arange(L-epsilon,H+2*epsilon,epsilon)
    M = len(price_list)
    
    V = np.zeros(M)
    mu = np.zeros(M)
    
    ####### parameters that can be tuned
    k = 1000 #number of iterations in policy evaluation, k could be any positive integer
    num_iters = 100 # numer of outermost loop
    converge_cnt = 0
    
    start_time = time.time()
    
    for count in range(num_iters):
        
        # policy improvement
        for i in range(M):
            V_candidate = np.zeros(M)
            for j in range(M):
                price_to_j = (price_list[j] - theta*price_list[i])/(1-theta)
                
                # unreachable state from state i
                if price_to_j > H or price_to_j < L:
                    V_candidate[j] = -np.inf
                else:
                    if func == 'mmnl':
                        V_candidate[j] = R(price_list[i], price_to_j \
                                           ,coefficients,weights) + gamma * V[j]
                    elif func == 'linear':
                        V_candidate[j] = R_lin(price_list[i],price_to_j \
                                           ,coefficients,weights) + gamma * V[j]
                    elif func == 'mmnl_ext':
                        V_candidate[j] = R_ext(price_list[i],price_to_j \
                                           ,coefficients,weights) + gamma * V[j]
#                    elif func == 'uniform':
#                        V_candidate[j] = R_uniform_fast(price_list[i],price_to_j,\
#                                  coefficients )+ gamma * V[j]
#                    elif func == 'mmnl_estimated':
#                        V_candidate[j] = R_mmnl_est(price_list[i],price_to_j,\
#                                  coefficients )+ gamma * V[j]
#                    elif func == 'logistic_estimated':
#                        V_candidate[j] = R_logistic_est(price_list[i],price_to_j,\
#                                  coefficients )+ gamma * V[j]
            id_max = np.argmax(np.round(V_candidate.ravel(), decimals_))
            # print(id_max)
            mu[i] = id_max

        # optimistic policy evaluation
        V_new = np.zeros(M)
        for i in range(M):
            ref_price_id = i
            for s in range(k):
                ref_price_id_new = int(mu[ref_price_id])
                price = (price_list[ref_price_id_new] - theta *price_list[ref_price_id])/(1-theta)
                if func == 'mmnl':
                    V_new[i] += gamma**s * R(price_list[ref_price_id], price\
                                           ,coefficients,weights )
                elif func == 'linear':
                    V_new[i] += gamma**s * R_lin(price_list[ref_price_id], price\
                          ,coefficients,weights ) 
                elif func == 'mmnl_ext':
                    V_new[i] += gamma**s *R_ext(price_list[ref_price_id],price \
                                           ,coefficients,weights) 
#                elif func == 'uniform':
#                    V_new[i] += gamma**s * R_uniform_fast(price_list[ref_price_id], price\
#                                           ,coefficients ) 
#                elif func == 'logistic_estimated':
#                    V_new[i] += gamma**s * R_logistic_est(price_list[ref_price_id], price\
#                                           ,coefficients ) 
                ref_price_id = ref_price_id_new
            V_new[i] += gamma**k * V[ref_price_id_new]
        
        # check convergence
        if np.linalg.norm(V-V_new,np.inf) <1e-100:
            converge_cnt +=1
        if converge_cnt >5:
            break
        V = np.copy(V_new)
        
    print('number of iterations: ',count+1)
    
    print("running time of pricing optimization {}".format(time.time()- start_time))
        
        #if not non_decreasing(V):
            #print('NOT non_decreasing')
    return V,mu

def myopic_inf_hor_pricing_pricerange(L,H,theta,epsilon,T,gamma,coefficients,weights,func):
    '''
    Input:
    L: lower bound of price
    H: upper bound of price
    theta: memory parameter of prices
    epsilon: accuracy of price discretization
    T: number of time periods
    gamma: discounted factor
    coefficients:  u = a - b * p + c_pos * np.maximum(r - p, 0) + c_neg * np.maximum(p - r, 0) utility model
    
    Output:
    uner myopic pricing policy
    V: V[i] = revenue for infinite horizon when the first (reference) price is price_list[i]
    mu: mu[i] = optimal next reference price in the next time period given reference price is price_list[i]
    '''
    if T != np.inf:
        raise ValueError("Must be infinite horizon!")
    
    k = 1000 #number of iterations in policy evaluation
    
    
    price_list = np.arange(L-epsilon,H+2*epsilon,epsilon)
    M = len(price_list)
    
    V = np.zeros(M)
    mu = np.zeros(M)
    
    for i in range(M):
        R_temp = np.zeros(M)
        for j in range(M):
            price_to_j = (price_list[j] - theta *price_list[i])/(1-theta)
            # unreachable state from state i
            if price_to_j > H or price_to_j < L:
                R_temp[j] = -np.inf
            else:
                if func == 'mmnl':
                    R_temp[j] = R(price_list[i], price_to_j,coefficients,weights)
                elif func == 'linear':
                    R_temp[j] = R_lin(price_list[i], price_to_j,coefficients,weights) 
                elif func == 'mmnl_ext':
                    R_temp[j] =  R_ext(price_list[i], price_to_j,coefficients,weights)
                
        mu[i] = np.argmax(R_temp)
    
    for i in range(M):
        ref_price_id = i
        for s in range(k):
            ref_price_id_new = int(mu[ref_price_id])
            price = (price_list[ref_price_id_new] - theta *price_list[ref_price_id])/(1-theta)
            if func == 'mmnl':
                V[i] += gamma**s * R(price_list[ref_price_id], price\
                         ,coefficients,weights )
            elif func == 'linear':
                V[i] += gamma**s * R_lin(price_list[ref_price_id], price\
                          ,coefficients,weights ) 
            elif func == 'mmnl_ext':
                V[i] += gamma**s * R_ext(price_list[ref_price_id], price\
                         ,coefficients,weights)
            ref_price_id = ref_price_id_new
    return V,mu

def policy_evaluation(L,H,theta,epsilon,T,gamma,coefficients,weights,func,mu):
    '''
    Evaluate the long term revenue of policy mu under the environment
    specified by (coefficents, weights, func)
    '''
    k = 1000 #number of iterations in policy evaluation
    
    price_list = np.arange(L-epsilon,H+2*epsilon,epsilon)
    M = len(price_list)
    
    V = np.zeros(M)
    
    for i in range(M):
        ref_price_id = i
        for s in range(k):
            ref_price_id_new = int(mu[ref_price_id])
            price = (price_list[ref_price_id_new] - theta *price_list[ref_price_id])/(1-theta)
            if func == 'mmnl':
                V[i] += gamma**s * R(price_list[ref_price_id], price\
                         ,coefficients,weights )
            elif func == 'linear':
                V[i] += gamma**s * R_lin(price_list[ref_price_id], price\
                          ,coefficients,weights ) 
            elif func == 'mmnl_ext':
                V[i] += gamma**s * R_ext(price_list[ref_price_id], price\
                         ,coefficients,weights)
            ref_price_id = ref_price_id_new
    return V
    
    
    
    
    
    
    
    
    
    
    
    
    

