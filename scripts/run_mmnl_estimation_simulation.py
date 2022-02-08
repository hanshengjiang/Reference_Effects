#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hanshengjiang
"""

'''
Run a simulation where the ground truth (mixing distribution)
 is a uniform distribution
 
Compare between NPMLE and logistic regression

under metrics of RMSE and MAE
'''

from utilis import *
from py_estimation import *
from mmnl_simulation import *


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from itertools import repeat
import multiprocessing 
from multiprocessing import Pool


    
def simulation_comparison(folder_id, num, T, T_test, params):
    
    np.random.seed(folder_id)
    
    timestamp = datetime.today().strftime('%m_%d_%Y')

    # record results
    npmle_insample = np.zeros((num,4)) # each row: (rmse, mae, kld)
    logit_insample = np.zeros((num,4))
    EM_insample = np.zeros((num,4))
    lm_insample = np.zeros((num,4))
    
    npmle_outsample = np.zeros((num,4))
    logit_outsample = np.zeros((num,4))
    EM_outsample = np.zeros((num,4))
    lm_outsample = np.zeros((num,4))
    
    for instance in tqdm(range(num)):
        
        # T = 500
        # T_test = 500 # size of testing set
        n = 2 # single product + no purchase product
        d = 4 # 1, p, (r-p)_+, (r-p)_-
        num_ppl = 1 # one people at one time
        
        
        #----training data----#
        V_alltime,choice_ct, prob_alltime, D_alltime = \
        simulated_uniform_data(T,n,d,num_ppl,params)
        

        # Gaussian
        # ### V_alltime,choice_ct = simulated_Gaussian_data(T,n,d,num_ppl)
        #---------------------#
        
        #------testing data----#
        V_test,choice_ct_test, prob_test, D_test = \
        simulated_uniform_data(T_test,n,d,num_ppl,params)
        #----------------------#
        
        pd.DataFrame(V_alltime[:,1,:]).to_csv('./../simulation_results/CGM/{}/V_alltime_{}_{}_{}_{}.csv'\
                    .format(folder_id,T,n,d,timestamp), index= False, header = False)
        pd.DataFrame(choice_ct).to_csv('./../simulation_results/CGM/{}/choice_ct_{}_{}_{}_{}.csv'\
                    .format(folder_id, T,n,d,timestamp), index= False, header = False)
        pd.DataFrame(V_test[:,1,:]).to_csv('./../simulation_results/CGM/{}/V_test_{}_{}_{}_{}.csv'\
                    .format(folder_id,T,n,d,timestamp), index = False, header = False)
        pd.DataFrame(choice_ct_test).to_csv('./../simulation_results/CGM/{}/choice_ct_test_{}_{}_{}_{}.csv'\
                    .format(folder_id,T,n,d,timestamp), index= False, header = False)
        
        
        #-----------------------------------------
        #----------------------------------------------------------------------------------
        # Linear model
        #----------------------------------------------------------------------------------
        #-----------------------------------------
        
        X = V_alltime[:,1,:]
        y = choice_ct[:,1]
        
        num_agg = 25 # number of points in each aggregation
        X_agg = np.zeros((int(X.shape[0]/num_agg), X.shape[1]))
        y_agg = np.zeros(int(X.shape[0]/num_agg))
        D_alltime1_agg = np.zeros(int(X.shape[0]/num_agg))
        
        for i in range(len(X_agg)):
            X_agg[i] = X[num_agg*i:num_agg*(i+1)].sum(axis = 0)/num_agg
            y_agg[i] = y[num_agg*i:num_agg*(i+1)].sum(axis = 0)
            D_alltime1_agg[i] = D_alltime[num_agg*i:num_agg*(i+1),1].sum(axis = 0)
#        
#        pd.DataFrame(X_agg).to_csv('./../simulation_results/Linear/{}/X_{}_{}_{}_{}.csv'\
#                    .format(folder_id,T,n,d,timestamp), index= False, header = False)
#        
#        pd.DataFrame(y_agg).to_csv('./../simulation_results/Linear/{}/y_{}_{}_{}_{}.csv'\
#                    .format(folder_id,T,n,d,timestamp), index = False, header = False)
#        
        reg = LinearRegression(
                fit_intercept = False
                ).fit(X_agg,y_agg)
        B_lm = reg.coef_
        alpha_lm = np.array([[1.0]])
        
        pd.DataFrame(B_lm).to_csv('./../simulation_results/Linear/{}/B_lm_{}_{}_{}_{}.csv'\
                     .format(folder_id,T,n,d,timestamp), index = False, header = False)
        pd.DataFrame(alpha_lm).to_csv('./../simulation_results/Linear/{}/alpha_lm_{}_{}_{}_{}.csv'\
                    .format(folder_id,T,n,d,timestamp), index = False, header = False)
        
        f_lm =  np.matmul(X_agg, B_lm)
        
        
        X_test = V_test[:,1,:]
        
        X_test_agg = np.zeros((int(X_test.shape[0]/num_agg), X_test.shape[1]))
        
        D_test1_agg = np.zeros(int(X_test.shape[0]/num_agg))
        
        for i in range(len(X_agg)):
            X_test_agg[i] = X_test[num_agg*i:num_agg*(i+1)].sum(axis = 0)/num_agg
            D_test1_agg[i] = D_test[num_agg*i:num_agg*(i+1),1].sum(axis = 0)
            
        f_lm_test = np.matmul(X_test_agg, B_lm)
        
        lm_insample[instance,0] = np.nan
        lm_insample[instance,1] = np.nan
        
        lm_insample[instance,2] = np.linalg.norm(f_lm-D_alltime1_agg, ord = 1)
        lm_insample[instance,3] = np.linalg.norm(f_lm-D_alltime1_agg, ord = 2)
       
        
        # testing
        lm_outsample[instance,0] = np.nan
        lm_outsample[instance,1] = np.nan
        
        lm_outsample[instance,2] = np.linalg.norm(f_lm_test-D_test1_agg, ord = 1)
        lm_outsample[instance,3] = np.linalg.norm(f_lm_test-D_test1_agg, ord = 2)

        
    
        
        #-----------------------------------------       
        #----------------------------------------------------------------------------------
        #    CGM
        #----------------------------------------------------------------------------------
        #-----------------------------------------
        
        # CGM(features, labels, iteration_times)
        
        # number of outer iterations of CGM
        num_CGMiter = 60
        
        f, B, vB,alpha, L_rec = CGM(V_alltime,choice_ct, num_CGMiter)
        
        #------------------------------------------------------
        # save estimation results of simulated data to files
        timestamp = datetime.today().strftime('%m_%d_%Y')
        pd.DataFrame(f).to_csv('./../simulation_results/CGM/{}/f_{}_{}_{}_{}.csv'.format(folder_id,T,n,d,timestamp), index= False, header = False )
        pd.DataFrame(B).to_csv('./../simulation_results/CGM/{}/B_{}_{}_{}_{}.csv'.format(folder_id,T,n,d,timestamp), index= False, header = False )
        pd.DataFrame(vB).to_csv('./../simulation_results/CGM/{}/vB_{}_{}_{}_{}.csv'.format(folder_id,T,n,d,timestamp), index= False, header = False )
        pd.DataFrame(alpha).to_csv('./../simulation_results/CGM/{}/alpha_{}_{}_{}_{}.csv'.format(folder_id,T,n,d,timestamp), index = False, header = False )
        pd.DataFrame(L_rec).to_csv('./../simulation_results/CGM/{}/L_rec_{}_{}_{}_{}.csv'.format(folder_id,T,n,d,timestamp), index = False, header = False )
        #------------------------------------------------------
        
        
        f_agg = np.zeros((int(f.shape[0]/num_agg), f.shape[1]))
        for i in range(len(f_agg)):
            f_agg[i] = f[num_agg*i:num_agg*(i+1)].sum(axis = 0)
            
 
        npmle_insample[instance,0] = np.linalg.norm(f[:,1]-D_alltime[:,1], ord = 1)
        npmle_insample[instance,1] = np.linalg.norm(f[:,1]-D_alltime[:,1], ord = 2)
        npmle_insample[instance,2] = np.linalg.norm(f_agg[:,1]-D_alltime1_agg, ord = 1)
        npmle_insample[instance,3] = np.linalg.norm(f_agg[:,1]-D_alltime1_agg, ord = 2)
        
        # testing
        f_test = predict_probability(B, vB, alpha, V_test)
        f_test_agg = np.zeros((int(f_test.shape[0]/num_agg), f_test.shape[1]))
        for i in range(len(f_agg)):
            f_test_agg[i] = f_test[num_agg*i:num_agg*(i+1)].sum(axis = 0)
            
        npmle_outsample[instance,0] = np.linalg.norm(f_test[:,1]-D_test[:,1], ord = 1)
        npmle_outsample[instance,1] = np.linalg.norm(f_test[:,1]-D_test[:,1], ord = 2)
        npmle_outsample[instance,2] = np.linalg.norm(f_test_agg[:,1]-D_test1_agg, ord = 1)
        npmle_outsample[instance,3] = np.linalg.norm(f_test_agg[:,1]-D_test1_agg, ord = 2)
        
        
        
        #-----------------------------------------
        #----------------------------------------------------------------------------------
        # mixed logit by EM
        #----------------------------------------------------------------------------------
        #-----------------------------------------
        
        X = V_alltime[:,1,:]
        y = choice_ct[:,1]
        
        # output data so that R can read it
        pd.DataFrame(X).to_csv('./../simulation_results/EM/{}/X.csv'.format(folder_id), index= False, header = False )
        pd.DataFrame(y).to_csv('./../simulation_results/EM/{}/y.csv'.format(folder_id), index = False, header = False)
        
        
        #-------------------------   Run EM    ----------------------#
        import os
        import sys
        
        
        k = B.shape[1] # keep number of components the same as NPMLE
        
        # random seed for R code
        r_seed = folder_id * num + instance
        
#        os.system("Rscript " + "run_logisregmixEM.R " + str(k) + " " + \
#                  str(r_seed) + " " + "./../simulation_results/EM/{}".format(str(folder_id)))
#        
        os.system("python " + "run_EM_KETrain.py " + str(k) + " " + str(r_seed) \
                          + " " + "./../simulation_results/EM/{}".format(str(folder_id)))
                          
        
        #-------------------Read EM estimated results ----------------#
        B_EM = pd.read_csv('./../simulation_results/EM/{}/B_EM.csv'.format(folder_id), header = None).values
        alpha_EM = pd.read_csv('./../simulation_results/EM/{}/alpha_EM.csv'.format(folder_id), header = None).values.ravel()
        #------------------------------------------------------------# 
        
        f_EM = predict_probability(B_EM, np.zeros(B_EM.shape), alpha_EM, V_alltime)
        f_EM_agg = np.zeros((int(f_EM.shape[0]/num_agg), f_EM.shape[1]))
        for i in range(len(f_EM_agg)):
            f_EM_agg[i] = f_EM[num_agg*i:num_agg*(i+1)].sum(axis = 0)
            
        
        
        EM_insample[instance,0] = np.linalg.norm(f_EM[:,1]-D_alltime[:,1], ord = 1)
        EM_insample[instance,1] = np.linalg.norm(f_EM[:,1]-D_alltime[:,1], ord = 2)
        EM_insample[instance,2] = np.linalg.norm(f_EM_agg[:,1]-D_alltime1_agg, ord = 1)
        EM_insample[instance,3] = np.linalg.norm(f_EM_agg[:,1]-D_alltime1_agg, ord = 2)
       
            
        # testing
        f_EM_test = predict_probability(B_EM, np.zeros(B_EM.shape), alpha_EM, V_test)
        f_EM_test_agg = np.zeros((int(f_EM_test.shape[0]/num_agg), f_EM_test.shape[1]))
        for i in range(len(f_EM_test_agg)):
            f_EM_test_agg[i] = f_EM_test[num_agg*i:num_agg*(i+1)].sum(axis = 0)
            
            
        EM_outsample[instance,0] = np.linalg.norm(f_EM_test[:,1]-D_test[:,1], ord = 1)
        EM_outsample[instance,1] = np.linalg.norm(f_EM_test[:,1]-D_test[:,1], ord = 2)
        EM_outsample[instance,2] = np.linalg.norm(f_EM_test_agg[:,1]-D_test1_agg, ord = 1)
        EM_outsample[instance,3] = np.linalg.norm(f_EM_test_agg[:,1]-D_test1_agg, ord = 2)
        
        
        
        #-----------------------------------------
        #----------------------------------------------------------------------------------
        # logistic regression by sklearn
        #----------------------------------------------------------------------------------
        #-----------------------------------------
        
        # reformat data for logistic regression
        X = V_alltime[:,1,:]
        y = choice_ct[:,1]
        
        if np.sum(choice_ct[:,1]) > 0 and np.sum(choice_ct[:,0]) > 0:
            clf = LogisticRegression(random_state = folder_id,
                                     multi_class = 'multinomial',
                                     solver = 'lbfgs',
                                     fit_intercept = False).fit(X,y)
           
            
            B_logit = np.reshape(clf.coef_, (d,1))
            alpha_logit = np.array([[1.0]])
        else:
            B_logit  = np.zeros((d,1))
            B_logit[:] = np.nan
            alpha_logit = np.array([[np.nan]])
            
        pd.DataFrame(B_logit).to_csv('./../simulation_results/Logit/{}/B_logit_{}_{}_{}_{}.csv'\
                    .format(folder_id,T,n,d,timestamp), index= False, header = False)
        pd.DataFrame(alpha_logit).to_csv('./../simulation_results/Logit/{}/alpha_logit_{}_{}_{}_{}.csv'\
                    .format(folder_id,T,n,d,timestamp), index = False, header = False)
        
        f_logit = predict_probability(B_logit, np.zeros(B_logit.shape), alpha_logit, V_alltime)
        
        f_logit_agg = np.zeros((int(f_logit.shape[0]/num_agg), f_logit.shape[1]))
        for i in range(len(f_logit_agg)):
            f_logit_agg[i] = f_logit[num_agg*i:num_agg*(i+1)].sum(axis = 0)
            
            
        logit_insample[instance,0] = np.linalg.norm(f_logit[:,1]-D_alltime[:,1], ord = 1)
        logit_insample[instance,1] = np.linalg.norm(f_logit[:,1]-D_alltime[:,1], ord = 2)
        logit_insample[instance,2] = np.linalg.norm(f_logit_agg[:,1]-D_alltime1_agg, ord = 1)
        logit_insample[instance,3] = np.linalg.norm(f_logit_agg[:,1]-D_alltime1_agg, ord = 2)
       
            
        # testing
        f_logit_test = predict_probability(B_logit, np.zeros(B_logit.shape), alpha_logit, V_test)
        
        f_logit_test_agg = np.zeros((int(f_logit_test.shape[0]/num_agg), f_logit_test.shape[1]))
        for i in range(len(f_logit_test_agg)):
            f_logit_test_agg[i] = f_logit_test[num_agg*i:num_agg*(i+1)].sum(axis = 0)
            
        logit_outsample[instance,0] = np.linalg.norm(f_logit_test[:,1]-D_test[:,1], ord = 1)
        logit_outsample[instance,1] = np.linalg.norm(f_logit_test[:,1]-D_test[:,1], ord = 2)
        logit_outsample[instance,2] = np.linalg.norm(f_logit_test_agg[:,1]-D_test1_agg, ord = 1)
        logit_outsample[instance,3] = np.linalg.norm(f_logit_test_agg[:,1]-D_test1_agg, ord = 2)
        
        
        
       
            
            
            
            
        # calculate final average metrics
        npmle_mae_in = np.sum(npmle_insample[:,0])/(T*num)
        print('npmle_mae_in ',npmle_mae_in )
        npmle_mae_out = np.sum(npmle_outsample[:,0])/(T_test*num)
        print('npmle_mae_out',npmle_mae_out)
        
        npmle_rmse_in = np.sqrt(np.sum(np.square(npmle_insample[:,1]))/(T*num))
        print('npmle_rmse_in',npmle_rmse_in)
        npmle_rmse_out = np.sqrt(np.sum(np.square(npmle_outsample[:,1]))/(T_test*num))
        print('npmle_rmse_out', npmle_rmse_out )
        
        npmle_mae_in_agg = np.sum(npmle_insample[:,2])/(T/num_agg*num)
        print('npmle_mae_in_agg ',npmle_mae_in_agg )
        npmle_mae_out_agg = np.sum(npmle_outsample[:,2])/(T_test/num_agg*num)
        print('npmle_mae_out_agg',npmle_mae_out_agg)
        
        npmle_rmse_in_agg = np.sqrt(np.sum(np.square(npmle_insample[:,3]))/(T/num_agg*num))
        print('npmle_rmse_in_agg',npmle_rmse_in_agg)
        npmle_rmse_out_agg = np.sqrt(np.sum(np.square(npmle_outsample[:,3]))/(T_test/num_agg*num))
        print('npmle_rmse_out_agg', npmle_rmse_out_agg )
    
        # calculate final average metrics
        EM_mae_in = np.sum(EM_insample[:,0])/(T*num)
        print('EM_mae_in',EM_mae_in)
        EM_mae_out = np.sum(EM_outsample[:,0])/(T_test*num)
        print('EM_mae_out',EM_mae_out)
        EM_rmse_in = np.sqrt(np.sum(np.square(EM_insample[:,1]))/(T*num))
        print('EM_rmse_in',EM_rmse_in)
        EM_rmse_out = np.sqrt(np.sum(np.square(EM_outsample[:,1]))/(T_test*num))
        print('EM_rmse_out',EM_rmse_out)
        
        EM_mae_in_agg = np.sum(EM_insample[:,2])/(T/num_agg*num)
        print('EM_mae_in_agg',EM_mae_in_agg)
        EM_mae_out_agg = np.sum(EM_outsample[:,2])/(T_test/num_agg*num)
        print('EM_mae_out_agg',EM_mae_out_agg)
        EM_rmse_in_agg = np.sqrt(np.sum(np.square(EM_insample[:,3]))/(T/num_agg*num))
        print('EM_rmse_in_agg',EM_rmse_in_agg)
        EM_rmse_out_agg = np.sqrt(np.sum(np.square(EM_outsample[:,3]))/(T_test/num_agg*num))
        print('EM_rmse_out_agg',EM_rmse_out_agg)

        
        
        # calculate final average metrics
        logit_mae_in = np.sum(logit_insample[:,0])/(T*num)
        print('logit_mae_in',logit_mae_in)
        logit_mae_out = np.sum(logit_outsample[:,0])/(T_test*num)
        print('logit_mae_out',logit_mae_out)
        logit_rmse_in = np.sqrt(np.sum(np.square(logit_insample[:,1]))/(T*num))
        print('logit_rmse_in',logit_rmse_in)
        logit_rmse_out = np.sqrt(np.sum(np.square(logit_outsample[:,1]))/(T_test*num))
        print('logit_rmse_out',logit_rmse_out)
    

        logit_mae_in_agg = np.sum(logit_insample[:,2])/(T/num_agg*num)
        print('logit_mae_in_agg',logit_mae_in_agg)
        logit_mae_out_agg = np.sum(logit_outsample[:,2])/(T_test/num_agg*num)
        print('logit_mae_out_agg',logit_mae_out_agg)
        logit_rmse_in_agg = np.sqrt(np.sum(np.square(logit_insample[:,3]))/(T/num_agg*num))
        print('logit_rmse_in_agg',logit_rmse_in_agg)
        logit_rmse_out_agg = np.sqrt(np.sum(np.square(logit_outsample[:,3]))/(T_test/num_agg*num))
        print('logit_rmse_out_agg',logit_rmse_out_agg)
        
        
        # calculate final average metrics

        lm_mae_in = np.sum(lm_insample[:,0])/(T*num)
        print('lm_mae_in',lm_mae_in)
        lm_mae_out = np.sum(lm_outsample[:,0])/(T_test*num)
        print('lm_mae_out',lm_mae_out)
        lm_rmse_in = np.sqrt(np.sum(np.square(lm_insample[:,1]))/(T*num))
        print('lm_rmse_in',lm_rmse_in)
        lm_rmse_out = np.sqrt(np.sum(np.square(lm_outsample[:,1]))/(T_test*num))
        print('lm_rmse_out',lm_rmse_out)
        
        lm_mae_in_agg = np.sum(lm_insample[:,2])/(T/num_agg*num)
        print('lm_mae_in_agg',lm_mae_in)
        lm_mae_out_agg = np.sum(lm_outsample[:,2])/(T_test/num_agg*num)
        print('lm_mae_out_agg',lm_mae_out)
        lm_rmse_in_agg = np.sqrt(np.sum(np.square(lm_insample[:,3]))/(T/num_agg*num))
        print('lm_rmse_in_agg',lm_rmse_in)
        lm_rmse_out_agg = np.sqrt(np.sum(np.square(lm_outsample[:,3]))/(T_test/num_agg*num))
        print('lm_rmse_out_agg',lm_rmse_out)
        
        return np.array([npmle_rmse_in,npmle_rmse_out, EM_rmse_in,EM_rmse_out, logit_rmse_in,logit_rmse_out, lm_rmse_in,lm_rmse_out,
  npmle_mae_in,npmle_mae_out, EM_mae_in,EM_mae_out, logit_mae_in,logit_mae_out, lm_mae_in,lm_mae_out,
  npmle_rmse_in_agg,npmle_rmse_out_agg, EM_rmse_in_agg,EM_rmse_out_agg, logit_rmse_in_agg,logit_rmse_out_agg, lm_rmse_in_agg,lm_rmse_out_agg,
  npmle_mae_in_agg,npmle_mae_out_agg, EM_mae_in_agg,EM_mae_out_agg, logit_mae_in_agg,logit_mae_out_agg, lm_mae_in_agg,lm_mae_out_agg])
    
name_array = ['npmle_rmse_in','npmle_rmse_out', 'EM_rmse_in','EM_rmse_out', 'logit_rmse_in','logit_rmse_out', 'lm_rmse_in','lm_rmse_out',
  'npmle_mae_in','npmle_mae_out', 'EM_mae_in','EM_mae_out', 'logit_mae_in','logit_mae_out', 'lm_mae_in','lm_mae_out',
  'npmle_rmse_in_agg','npmle_rmse_out_agg', 'EM_rmse_in_agg','EM_rmse_out_agg', 'logit_rmse_in_agg','logit_rmse_out_agg', 'lm_rmse_in_agg','lm_rmse_out_agg',
  'npmle_mae_in_agg','npmle_mae_out_agg', 'EM_mae_in_agg','EM_mae_out_agg', 'logit_mae_in_agg','logit_mae_out_agg', 'lm_mae_in_agg','lm_mae_out_agg']
    
#--------------
params = (1,3,0,3,10) #distributions of parameter and coefficients 
num = 5
T = 500
T_test = 500
#--------------

parallel_n = 20 # total number of instances = num * parallel_n


# np.random.seed(626)

with Pool(20) as sim_pool:
    metric_results = sim_pool.starmap(simulation_comparison, zip(np.arange(1,parallel_n+1),\
                                                                 repeat(num),repeat(T),repeat(T_test),repeat(params)))
    print(metric_results)
    metric_results_arr = np.zeros((parallel_n,32))
    for i in range(parallel_n):
        metric_results_arr[i] = np.array(metric_results[i]).ravel()
    pd.DataFrame(metric_results_arr).to_csv('./../simulation_results/metric_results_1.csv', index= False, header = False )   
    

#--------------------------------------------------------
# summarize results from parallel computing
# --------------------------------------------------------

metric_results_arr = pd.read_csv('./../simulation_results/metric_results.csv', header = None).values

metric_summary = np.zeros(32)

for j in range(32):
    if j<8 or (j >= 16 and j < 24):
        metric_summary[j] = np.sqrt(np.sum(np.square(metric_results_arr[:,j]))/(parallel_n))
    else:
        metric_summary[j] = np.sum(metric_results_arr[:,j])/(parallel_n)
    print(name_array[j], "{:.3f}".format(metric_summary[j]))
pd.DataFrame(metric_summary,index= name_array).to_csv('./../simulation_results/metric_summary.csv', header = False)

#-----------------------------------------------------------
metric_summary = pd.read_csv('./../simulation_results/metric_summary.csv', header = None).values[:,1] # ignore index

# reformulate the metric so that it is in the same format as the table in paper
metric_latex = np.zeros((4,8))
metric_latex[0] = metric_summary[[0,8,1,9,16,24,17,25]]
metric_latex[1] = metric_summary[[2,10,3,11,18,26,19,27]]
metric_latex[2] = metric_summary[[4,12,5,13,20,28,21,29]]
metric_latex[3] = metric_summary[[6,14,7,15,22,30,23,31]]     
pd.DataFrame(metric_latex).to_csv('./../simulation_results/metric_latex.csv', index = False, header = False)
    
    
    
    
    
    
    
    
    
    
    
    
        
        