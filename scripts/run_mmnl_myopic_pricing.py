#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: hanshengjiang
"""

from multiprocessing import Pool
from optimal_pricing_policy_exp_update import *
def JD_myopic_pricing(sku_ID, L,H, sub_sampling):
    
     # timestamp = datetime.today().strftime('%m_%d_%Y')
    timestamp = "01_10_2022"
    
    
    theta = 0.8
    # discretization accuracy
    epsilon = (H-L)/500
    
    gamma = 0.95
    
    price_list = np.arange(L-epsilon,H+2*epsilon,epsilon)
    M = len(price_list)
    T = np.inf
        
    # estimation results produced by "run_mmnl_estimation_simulation.py"
    B = pd.read_csv('./../MSOM_data_estimated/{}/CGM/B_{}.csv'.format(sku_ID,timestamp), header = None).values
    vB = pd.read_csv('./../MSOM_data_estimated/{}/CGM/vB_{}.csv'.format(sku_ID,timestamp), header = None).values
    alpha = pd.read_csv('./../MSOM_data_estimated/{}/CGM/alpha_{}.csv'.format(sku_ID,timestamp), header = None).values
    
    # processing
    alpha = alpha.ravel()
    BvB = np.append(B,vB, axis = 0)
    coefficients = (BvB.T).ravel()
    weights = alpha
    demand_name = 'mmnl_ext'
    
    V_npmle,mu_npmle \
    = inf_hor_pricing_pricerange(L,H,theta,epsilon,np.inf,gamma,\
                                 coefficients,weights,demand_name)
    npmle_mean = np.sum(V_npmle)/M
    npmle_median = np.median(V_npmle.ravel())
    

        
    # myopic pricing policy
    # myopic with respect to true model
    V_m,mu_m = myopic_inf_hor_pricing_pricerange(L,H,theta,epsilon,T,gamma,coefficients,weights,demand_name)

    pd.DataFrame(V_m).to_csv('./../MSOM_data_estimated/{}/V_myopic'.format(sku_ID), index = False, header = False)
    pd.DataFrame(mu_m).to_csv('./../MSOM_data_estimated/{}/mu_myopic'.format(sku_ID), index = False, header = False)
    
    myopic_mean = np.sum(V_m)/M
    myopic_median = np.median(V_m.ravel())
    
    print(sku_ID, "npmle_mean:", npmle_mean, "myopic_mean:", myopic_mean, \
          "npmle_median:", npmle_median, "myopic_median:", myopic_median)
    
count = 2000
with Pool(3) as pool:
    pool.starmap(JD_myopic_pricing, \
                    [('adfedb6893',50,160,count),\
                     ('3c79df1d80',30,60,count),\
                     ('b5cb479f7d', 20,50,count) ])
    
    