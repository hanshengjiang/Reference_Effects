#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hanshengjiang
"""

from itertools import repeat
import multiprocessing 

from datetime import datetime
from multiprocessing import Pool
from optimal_pricing_policy_exp_update import *

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


#----------------------------revenue-------------------------------------------
def JD_revenue_compare(sku_ID, L,H, sub_sampling, timestamp, ground_truth = "npmle"): 
    

    
    for theta in [0.8]:
        
        # discretization accuracy
        epsilon = (H-L)/500
        
        gamma = 0.95
        # theta = 0.8
        
        
        #-----------------------------------------       
        #----------------------------------------------------------------------------------
        #    Linear model
        #----------------------------------------------------------------------------------
        #-----------------------------------------
        
        coefficients = pd.read_csv('./../MSOM_data_estimated/{}/Linear/B_lm_{}.csv'\
                     .format(sku_ID, timestamp), header = None).values.ravel()

        weights = np.array([1.0])
        demand_name = 'linear'
        
        
        V_linear,mu_linear \
        = inf_hor_pricing_pricerange(L,H,theta,epsilon,np.inf,gamma,\
                                 coefficients,weights,demand_name)
        # save to file
        pd.DataFrame(V_linear).to_csv('./../MSOM_data_estimated/{}/V_linear'.format(sku_ID), index = False, header = False)
        pd.DataFrame(mu_linear).to_csv('./../MSOM_data_estimated/{}/mu_linear'.format(sku_ID), index = False, header = False)
    
    
        #-----------------------------------------       
        #----------------------------------------------------------------------------------
        #    CGM
        #----------------------------------------------------------------------------------
        #-----------------------------------------
        
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
        
        # save to file
        pd.DataFrame(V_npmle).to_csv('./../MSOM_data_estimated/{}/V_npmle'.format(sku_ID), index = False, header = False)
        pd.DataFrame(mu_npmle).to_csv('./../MSOM_data_estimated/{}/mu_npmle'.format(sku_ID), index = False, header = False)
    
        
        #-----------------------------------------
        #----------------------------------------------------------------------------------
        # mixed logit by EM
        #----------------------------------------------------------------------------------
        #-----------------------------------------
    
     
        #-------------------Read EM estimated results ----------------#
        B_EM = pd.read_csv('./../MSOM_data_estimated/{}/EM/B_EM.csv'.format(sku_ID), header = None).values
        alpha_EM = pd.read_csv('./../MSOM_data_estimated/{}/EM/alpha_EM.csv'.format(sku_ID), header = None).values.ravel()
        #------------------------------------------------------------# 
        demand_name = 'mmnl'
        coefficients = (B_EM.T).ravel()
        weights = alpha_EM.ravel()
        
        V_EM,mu_EM \
        = inf_hor_pricing_pricerange(L,H,theta,epsilon,np.inf,gamma,\
                                     coefficients,weights,demand_name)
        
        # save to file
        pd.DataFrame(V_EM).to_csv('./../MSOM_data_estimated/{}/V_EM'.format(sku_ID), index = False, header = False)
        pd.DataFrame(mu_EM).to_csv('./../MSOM_data_estimated/{}/mu_EM'.format(sku_ID), index = False, header = False)
    
        
        #-----------------------------------------
        #----------------------------------------------------------------------------------
        # logistic regression by sklearn
        #----------------------------------------------------------------------------------
        #-----------------------------------------
        
        #-------------------Read logit estimated results ----------------#
        B_logit = pd.read_csv('./../MSOM_data_estimated/{}/Logit/B_logit.csv'.format(sku_ID), header = None).values
        alpha_logit = pd.read_csv('./../MSOM_data_estimated/{}/Logit/alpha_logit.csv'.format(sku_ID), header = None).values.ravel()
        #------------------------------------------------------------# 
        demand_name = 'mmnl'
        coefficients = (B_logit.T).ravel()
        weights = alpha_logit.ravel()
        
        V_logit,mu_logit \
        = inf_hor_pricing_pricerange(L,H,theta,epsilon,np.inf,gamma,\
                                     coefficients,weights,demand_name)
        # save to file
        pd.DataFrame(V_logit).to_csv('./../MSOM_data_estimated/{}/V_logit'.format(sku_ID), index = False, header = False)
        pd.DataFrame(mu_logit).to_csv('./../MSOM_data_estimated/{}/mu_logit'.format(sku_ID), index = False, header = False)
    
    
        
        
        
    
        
        #----------------
        # Policy evaluation
        #----------------
        
        price_list = np.arange(L-epsilon,H+2*epsilon,epsilon)
        M = len(price_list)
        T = np.inf
        
        
        if ground_truth == "npmle":
            # use npmle as ground truth
            func = 'mmnl_ext'
            coefficients = (BvB.T).ravel()
            weights = alpha
        elif ground_truth == "EM":
            func = 'mmnl'
            coefficients = (B_EM.T).ravel()
            weights = alpha_EM.ravel()
        elif ground_truth == "logit":
            func = 'mmnl'
            coefficients = (B_logit.T).ravel()
            weights = alpha_logit.ravel()
        elif ground_truth == "linear":
            func = "linear"
            coefficients = pd.read_csv('./../MSOM_data_estimated/{}/Linear/B_lm_{}.csv'\
                     .format(sku_ID, timestamp), header = None).values.ravel()
            weights = np.array([1.0])
        
       
        
        V = policy_evaluation(L,H,theta,epsilon,T,gamma,coefficients,weights,func,mu_npmle)
        npmle_mean = np.sum(V)/M
        npmle_median = np.median(V.ravel())
        
        V = policy_evaluation(L,H,theta,epsilon,T,gamma,coefficients,weights,func,mu_EM)
        EM_mean = np.sum(V)/M
        EM_median = np.median(V.ravel())
        
        
        V = policy_evaluation(L,H,theta,epsilon,T,gamma,coefficients,weights,func,mu_logit)
        logit_mean = np.sum(V)/M
        logit_median = np.median(V.ravel())
        pd.DataFrame(V).to_csv('./../MSOM_data_estimated/{}/V_logit_under{}'\
                                 .format(sku_ID, func), index = False, header = False)
     
    
        V = policy_evaluation(L,H,theta,epsilon,T,gamma,coefficients,weights,func,mu_linear)
        linear_mean = np.sum(V)/M
        linear_median = np.median(V.ravel())
    
        
        # myopic pricing policy
        # myopic with respect to true model
        V_m,mu_m = myopic_inf_hor_pricing_pricerange(L,H,theta,epsilon,T,gamma,coefficients,weights,'mmnl_ext')
    
        pd.DataFrame(V_m).to_csv('./../MSOM_data_estimated/{}/V_myopic'.format(sku_ID), index = False, header = False)
        pd.DataFrame(mu_m).to_csv('./../MSOM_data_estimated/{}/mu_myopic'.format(sku_ID), index = False, header = False)
    
        # print results
        myopic_mean = np.sum(V_m)/M
        myopic_median = np.median(V_m.ravel())
        print(sku_ID, npmle_mean, myopic_mean, npmle_median,myopic_median)
        
        
        return np.array([npmle_mean,  EM_mean, logit_mean, linear_mean,\
                         npmle_median,EM_median,logit_median,linear_median])

count = 2000
import sys
if __name__ == "__main__":
    if len(sys.argv) < 2:
        ground_truth = "npmle"
    else:  
        #sys_argv[0] is the name of the .py file
        ground_truth = sys.argv[1]
    
with Pool(3) as pool:
    # timestamp = datetime.today().strftime('%m_%d_%Y')
    timestamp = "01_16_2022"
    
    jd_revenue_results = pool.starmap(JD_revenue_compare, \
                    [('adfedb6893',50,160,count, timestamp, ground_truth),\
                     ('3c79df1d80',30,60,count, timestamp, ground_truth),\
                     ('b5cb479f7d', 20,50,count, timestamp, ground_truth) ])
    #print(jd_revenue_results)
    jd_revenue_results_arr = np.zeros((3,8))
    for i in range(3):
        jd_revenue_results_arr[i] = np.array(jd_revenue_results[i]).ravel()
    pd.DataFrame(jd_revenue_results_arr).to_csv('./../MSOM_data_optimized/jd_revenue_results_{}_{}.csv'.format(timestamp,ground_truth),\
                index= False, header = False )   
    
    # reformulate into the same format as the table in paper
    jd_revenue_latex = np.zeros((4,6))
    jd_revenue_latex[:,0] = jd_revenue_results_arr[0,:4]
    jd_revenue_latex[:,1] = jd_revenue_results_arr[0,4:]
    jd_revenue_latex[:,2] = jd_revenue_results_arr[1,:4]
    jd_revenue_latex[:,3] = jd_revenue_results_arr[1,4:]
    jd_revenue_latex[:,4] = jd_revenue_results_arr[2,:4]
    jd_revenue_latex[:,5] = jd_revenue_results_arr[2,4:]
    
    pd.DataFrame(jd_revenue_latex).to_csv('./../MSOM_data_optimized/jd_revenue_latex_{}_{}.csv'.format(timestamp,ground_truth),\
                index= False, header = False ) 










