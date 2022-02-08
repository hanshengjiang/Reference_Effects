#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hanshengjiang
"""

'''
Compare linear model and logit model
in prediction accuracy and revenue performances

'''
from py_estimation import *
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from itertools import repeat
import multiprocessing 
from multiprocessing import Pool


#-------- purchase ratios-------------------------------#



def JD_prediction_compare(sku_ID, L,H, sub_sampling, timestamp, pooled_metric = True): 
    

    
    num_days = 31 # total number of days in March 2018
    
    for theta in [0.8]:
        
        # read real data
        df = pd.read_csv('./../MSOM_data_cleaned/%s_%s_V_daily'%(sku_ID,str(theta)[:3].replace('.', 'dot')))
        V_daily = df.values
        
        N = len(V_daily)
        
        # process readed data, prepare for estimation
        V_alltime = np.zeros((N, 2, 4))
        V_alltime[:,0,:] = 0 # no purchase feature
        V_alltime[:,1,:] = V_daily[:,1:5]
        
        # change sign of price features (1, -p, (r-p)_+, (r-p)_-)
        V_alltime[:,1,1] = - V_alltime[:,1,1]
        
        # choice data
        choice_ct = np.zeros((N, 2))
        # two "products" - purchase and no-purchase
        choice_ct[:,0] = np.maximum(1 - V_daily[:,5].ravel(),0)
        choice_ct[:,1] = np.minimum(1, V_daily[:,5].ravel())
        #---------------------------------------#
        
        D_alltime1_agg_total = np.zeros((num_days,1))
        num_ppl_arr_total = np.zeros(num_days)
        for i in range(num_days):
            D_alltime1_agg_total[i] = choice_ct[V_daily[:,0] == i+1,1].sum(axis = 0)
            num_ppl_arr_total[i] = len(choice_ct[V_daily[:,0] == i+1,1])
        pooled_proba_total = D_alltime1_agg_total.sum() / num_ppl_arr_total.sum()
        
        
        
        #-----------------------------------------       
        #----------------------------------------------------------------------------------
        #    CGM
        #----------------------------------------------------------------------------------
        #-----------------------------------------
        
        if len(choice_ct) > sub_sampling:
            V_alltime, choice_ct, V_daily = sub_sample_daily(V_alltime, choice_ct, V_daily, sub_sampling)
        
        D_alltime1_agg = np.zeros((num_days,1))
        num_ppl_arr = np.zeros(num_days)
        for i in range(num_days):
            D_alltime1_agg[i] = choice_ct[V_daily[:,0] == i+1,1].sum(axis = 0)
            num_ppl_arr[i] = len(choice_ct[V_daily[:,0] == i+1,1])
        pooled_proba = D_alltime1_agg.sum() / num_ppl_arr.sum()

        
        # try if file already exists
        from pathlib import Path
        my_file = Path('./../MSOM_data_estimated/{}/CGM/f_{}.csv'.format(sku_ID, timestamp))
        

        if my_file.is_file():
            f = pd.read_csv('./../MSOM_data_estimated/{}/CGM/f_{}.csv'.format(sku_ID, timestamp), header = None).values
            B = pd.read_csv('./../MSOM_data_estimated/{}/CGM/B_{}.csv'.format(sku_ID,timestamp), header = None).values
            vB = pd.read_csv('./../MSOM_data_estimated/{}/CGM/vB_{}.csv'.format(sku_ID,timestamp), header = None ).values
            alpha = pd.read_csv('./../MSOM_data_estimated/{}/CGM/alpha_{}.csv'.format(sku_ID,timestamp), header = None).values
            L_rec = pd.read_csv('./../MSOM_data_estimated/{}/CGM/L_rec_{}.csv'.format(sku_ID,timestamp), header = None ).values
        
        else:
            # print("File Not Found: " + './../MSOM_data_estimated/{}/CGM/f_{}.csv'.format(sku_ID, timestamp))
            start_time = time.time()
            f, B, vB, alpha, L_rec = CGM(V_alltime, choice_ct, 100)
            print("--- %s seconds ---" % (time.time() - start_time))
            
            #------------------------------------------------------
            # save estimation results of simulated data to files
            pd.DataFrame(f).to_csv('./../MSOM_data_estimated/{}/CGM/f_{}.csv'.format(sku_ID, timestamp), index= False, header = False )
            pd.DataFrame(B).to_csv('./../MSOM_data_estimated/{}/CGM/B_{}.csv'.format(sku_ID,timestamp), index= False, header = False )
            pd.DataFrame(vB).to_csv('./../MSOM_data_estimated/{}/CGM/vB_{}.csv'.format(sku_ID,timestamp), index= False, header = False )
            pd.DataFrame(alpha).to_csv('./../MSOM_data_estimated/{}/CGM/alpha_{}.csv'.format(sku_ID,timestamp), index = False, header = False )
            pd.DataFrame(L_rec).to_csv('./../MSOM_data_estimated/{}/CGM/L_rec_{}.csv'.format(sku_ID,timestamp), index = False, header = False )
            #------------------------------------------------------
            
        # save to one file
        #------------------------------------------------------
        alphaBvB = np.append(np.reshape(alpha.ravel(), (len(alpha),1)),\
                         (np.append(B,vB, axis = 0)).T, axis = 1)
        idx = (-alpha.ravel()).argsort()
        alphaBvB =  alphaBvB[idx]
    
        pd.DataFrame(alphaBvB).to_csv('./../MSOM_data_estimated/{}/CGM/alphaBvB_{}.csv'.format(sku_ID,timestamp),\
                     index = False, header = False, float_format='%.3f')
        #------------------------------------------------------
        
        
        f_agg_test = np.zeros((num_days, f.shape[1]))
        for i in range(len(f_agg_test)):
            f_agg_test[i] = f[V_daily[:,0] == i+1].sum(axis = 0)
        
        if pooled_metric == False:
            npmle_rmse = np.linalg.norm(f_agg_test[:,1]/num_ppl_arr-D_alltime1_agg/num_ppl_arr, ord = 2)/np.sqrt(num_days)
            npmle_mae = np.linalg.norm(f_agg_test[:,1]/num_ppl_arr-D_alltime1_agg/num_ppl_arr, ord = 1)/num_days
        else:
            npmle_rmse = np.linalg.norm(f_agg_test[:,1]/num_ppl_arr-pooled_proba, ord = 2)/np.sqrt(num_days) 
            npmle_mae = np.linalg.norm(f_agg_test[:,1]/num_ppl_arr-pooled_proba, ord = 1)/num_days
        
        
        
        #-----------------------------------------
        #----------------------------------------------------------------------------------
        # logistic regression by sklearn
        #----------------------------------------------------------------------------------
        #-----------------------------------------
        
        # random seed 
        r_seed = ord(sku_ID[0])
        d = 4
        
        # reformat data for logistic regression
        X = V_alltime[:,1,:]
        y = choice_ct[:,1]
        
        if np.sum(choice_ct[:,1]) > 0 and np.sum(choice_ct[:,0]) > 0:
            clf = LogisticRegression(random_state = r_seed,
                                     multi_class = 'multinomial',
                                     solver = 'lbfgs',
                                     fit_intercept = False).fit(X,y)
           
            
            B_logit = np.reshape(clf.coef_, (d,1))
            alpha_logit = np.array([1.0])
        else:
            B_logit  = np.zeros((d,1))
            B_logit[:] = np.nan
            alpha_logit = np.array([[np.nan]])
            
        pd.DataFrame(B_logit).to_csv('./../MSOM_data_estimated/{}/Logit/B_logit_{}.csv'\
                    .format(sku_ID,timestamp), index= False, header = False)
        pd.DataFrame(alpha_logit).to_csv('./../MSOM_data_estimated/{}/Logit/alpha_logit_{}.csv'\
                    .format(sku_ID,timestamp), index = False, header = False)
        
        f_logit = predict_probability(B_logit, np.zeros(B_logit.shape), alpha_logit, V_alltime)
        
        pd.DataFrame(f_logit).to_csv('./../MSOM_data_estimated/{}/Logit/f_logit_{}.csv'\
                                     .format(sku_ID, timestamp), index= False, header = False )
        
        f_logit_agg = np.zeros((num_days, f_logit.shape[1]))
        for i in range(len(f_logit_agg)):
            f_logit_agg[i] = f_logit[V_daily[:,0] == i+1].sum(axis = 0)
        
        if pooled_metric == False:
            logit_rmse = np.linalg.norm(f_logit_agg[:,1]/num_ppl_arr-D_alltime1_agg/num_ppl_arr, ord = 2)/np.sqrt(num_days)
            logit_mae = np.linalg.norm(f_logit_agg[:,1]/num_ppl_arr-D_alltime1_agg/num_ppl_arr, ord = 1)/num_days
        else: 
            logit_rmse = np.linalg.norm(f_logit_agg[:,1]/num_ppl_arr-pooled_proba, ord = 2)/np.sqrt(num_days)
            logit_mae = np.linalg.norm(f_logit_agg[:,1]/num_ppl_arr-pooled_proba, ord = 1)/num_days
        
        
         
        #-----------------------------------------
        #----------------------------------------------------------------------------------
        # mixed logit by EM
        #----------------------------------------------------------------------------------
        #-----------------------------------------
        
        
        # normalize feature matrix otherwise EM fails due to numerical underflow
        V_alltime_normed = np.copy(V_alltime)
        V_alltime_normed[:,1,:] = V_alltime[:,1,:]/(2*V_alltime[:,1,:].max(axis = 0)+1e-30) # avoid really small numbers
        
        
        X = V_alltime[:,1,:]
        y = choice_ct[:,1]
        
        # output data so that R can read it
        pd.DataFrame(X).to_csv("./../MSOM_data_estimated/{}/EM/X.csv".format(sku_ID), index= False, header = False )
        pd.DataFrame(y).to_csv("./../MSOM_data_estimated/{}/EM/y.csv".format(sku_ID), index = False, header = False)
        
        # save estiamtion results to be used as initialization for EM
        pd.DataFrame(B_logit).to_csv('./../MSOM_data_estimated/{}/EM/B_logit.csv'\
                    .format(sku_ID), index= False, header = False)
        pd.DataFrame(alpha_logit).to_csv('./../MSOM_data_estimated/{}/EM/alpha_logit.csv'\
                    .format(sku_ID), index = False, header = False)
        
        
        try:
            #-------------------------   Run EM    ----------------------#
            import os
            import sys
            
            k = B.shape[1] # EM easily fail with larger number of components
            
            my_file_EM = Path('./../MSOM_data_estimated/{}/EM/B_EM.csv'.format(sku_ID))
            
            if my_file_EM.is_file() == False:
                # option 1: run R package mixtools
                # this option is likely to fail due to numerical underflow
    #            os.system("Rscript " + "run_logisregmixEM.R " + str(k) + " " + str(r_seed) \
    #                      + " " + "./../MSOM_data_estimated/{}/EM".format(sku_ID)
    #                      )
                # option 2: run self-written Python code
                os.system("python " + "run_EM_KETrain.py " + str(k) + " " + str(r_seed) \
                          + " " + "./../MSOM_data_estimated/{}/EM".format(sku_ID)
                          )
    
            
            #-------------------Read EM estimated results ----------------#
            B_EM = pd.read_csv('./../MSOM_data_estimated/{}/EM/B_EM.csv'.format(sku_ID), header = None).values
            alpha_EM = pd.read_csv('./../MSOM_data_estimated/{}/EM/alpha_EM.csv'.format(sku_ID), header = None).values.ravel()
            #------------------------------------------------------------# 
            
            
            f_EM = predict_probability(B_EM, np.zeros(B_EM.shape), alpha_EM, V_alltime)
            
            pd.DataFrame(f_EM).to_csv('./../MSOM_data_estimated/{}/EM/f_EM_{}.csv'\
                                         .format(sku_ID, timestamp), index= False, header = False )
                
            f_EM_agg = np.zeros((num_days, f_EM.shape[1]))
            for i in range(len(f_EM_agg)):
                f_EM_agg[i] = f_EM[V_daily[:,0] == i+1].sum(axis = 0)
            
            if pooled_metric == False:
                EM_rmse = np.linalg.norm(f_EM_agg[:,1]/num_ppl_arr-D_alltime1_agg/num_ppl_arr, ord = 2)/np.sqrt(num_days)
                EM_mae = np.linalg.norm(f_EM_agg[:,1]/num_ppl_arr-D_alltime1_agg/num_ppl_arr, ord = 1)/num_days 
            else:
                EM_rmse = np.linalg.norm(f_EM_agg[:,1]/num_ppl_arr-pooled_proba, ord = 2)/np.sqrt(num_days)
                EM_mae = np.linalg.norm(f_EM_agg[:,1]/num_ppl_arr-pooled_proba, ord = 1)/num_days    
        except FileNotFoundError:
            print("EM fails")
            EM_rmse = np.nan
            EM_mae = np.nan
        
        #-----------------------------------------       
        #----------------------------------------------------------------------------------
        #    Linear model
        #----------------------------------------------------------------------------------
        #-----------------------------------------
        
        # from a different data source
        V_lin_df = pd.read_csv('./../MSOM_data_cleaned/%s_%s_V_lin'%(sku_ID,str(theta).replace('.', 'dot')))
        V_lin = V_lin_df.values
        D_df = pd.read_csv('./../MSOM_data_cleaned/%s_%s_D'%(sku_ID,str(theta).replace('.', 'dot')))
        D = D_df.values
        
        
        order_qty = np.sum(D)/np.sum(num_ppl_arr)
        D = D/order_qty # normalize the order quantity to align with purchase ratios
        
        # do NOT add extra constant intercept 
        reg = LinearRegression(
                fit_intercept = False
                ).fit(V_lin, D)
        
        B_lm = reg.coef_
        alpha_lm = np.array([[1.0]])
        
        pd.DataFrame(B_lm).to_csv('./../MSOM_data_estimated/{}/Linear/B_lm_{}.csv'\
                     .format(sku_ID, timestamp), index = False, header = False)
        pd.DataFrame(alpha_lm).to_csv('./../MSOM_data_estimated/{}/Linear/alpha_lm_{}.csv'\
                    .format(sku_ID, timestamp), index = False, header = False)
        
        
        f_lm =  np.matmul(V_lin, B_lm.T)
        
       
        if pooled_metric == False:
            linear_rmse = np.linalg.norm(f_lm/num_ppl_arr_total - D_alltime1_agg_total/num_ppl_arr_total, ord = 2)/np.sqrt(num_days)
            linear_mae = np.linalg.norm(f_lm/num_ppl_arr_total - D_alltime1_agg_total/num_ppl_arr_total, ord = 1)/num_days      
        else: 
            linear_rmse = np.linalg.norm(f_lm/num_ppl_arr_total - pooled_proba_total, ord = 2)/np.sqrt(num_days)
            linear_mae = np.linalg.norm(f_lm/num_ppl_arr_total - pooled_proba_total, ord = 1)/num_days         
        
        

        return np.array([npmle_rmse, npmle_mae, EM_rmse, EM_mae, logit_rmse, logit_mae, linear_rmse, linear_mae])

count = 2000

with Pool(3) as pool:
    
    # timestamp = datetime.today().strftime('%m_%d_%Y')
    
    timestamp = '02_05_2022'
    
    jd_metric_results = pool.starmap(JD_prediction_compare, \
            [('adfedb6893',50,160,count,timestamp),('3c79df1d80',30,60,count,timestamp),\
             ('b5cb479f7d', 20,50,count,timestamp) ])
    
    # print(jd_metric_results)
    jd_metric_results_arr = np.zeros((4,6))
    
    # save to file in the same format as the table in paper
    for i in range(4):
        jd_metric_results_arr[i] = np.array(jd_metric_results)[:,2*i:2*i+2].ravel()
    pd.DataFrame(jd_metric_results_arr).to_csv('./../MSOM_data_estimated/jd_metric_results_{}.csv'.format(timestamp), \
                index= False, header = False )   
    print(jd_metric_results_arr)
    
    
    
    
    
    
    
    
    
    
    
    

'''
#---------------------------test of endogeneity by correlation-----------------#
for theta in [0.8]:
    for (sku_ID, L,H, sub_sampling) in [('adfedb6893',50,160,10000),('3c79df1d80',30,60,10000),('7e4cb4952a', 20, 50,10000),('b5cb479f7d', 20,50,10000) ]:    
        li = pd.read_csv('./MSOM_data_estimated/{}{}_{}_{}_compare_demand'.\
                            format(option,sku_ID,str(theta)[:3].replace('.', 'dot'),str(sub_sampling)))
        li_arr = np.array(li)
        #calculate daily avaerage price
        V_lin_df = pd.read_csv('./MSOM_data_cleaned/%s_%s_V_lin'%(sku_ID,str(theta).replace('.', 'dot')))
        V_lin = V_lin_df.values
        li_price = np.zeros(day_end-day_start)
        for day in np.arange(day_start,day_end):
            li_price[day] = np.reshape(V_lin[day,:],(4,))[1]
        true = np.divide(li_arr[:,2],li_arr[:,1])
        mmnl = np.divide(li_arr[:,5],li_arr[:,1])
        linear = np.divide(li_arr[:,3],li_arr[:,1])
        # calculate correlation coefficient
        print('correlation mmnl', np.corrcoef(true-mmnl, li_price))
        print('correlation linear', np.corrcoef(true- linear, li_price))
     
# sample output of coefficient
#correlation mmnl [[ 1.         -0.13836073]
# [-0.13836073  1.        ]]
#correlation linear [[ 1.         -0.16113486]
# [-0.16113486  1.        ]]
#correlation mmnl [[ 1.         -0.20662647]
# [-0.20662647  1.        ]]
#correlation linear [[1.         0.10518908]
# [0.10518908 1.        ]]
#correlation mmnl [[ 1.         -0.30038191]
# [-0.30038191  1.        ]]
#correlation linear [[1.         0.08042882]
# [0.08042882 1.        ]]
#correlation mmnl [[ 1.         -0.01647397]
# [-0.01647397  1.        ]]
#correlation linear [[1.         0.24736375]
# [0.24736375 1.        ]]
'''           



        
        
        