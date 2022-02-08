#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 00:55:36 2020

@author: hanshengjiang
"""

from py_estimation import *
from py_MSOM_cleaning import *
from sklearn.linear_model import LogisticRegression
 
np.random.seed(626)
random.seed(626)

sku_ID = 'adfedb6893'
# '3c79df1d80','7e4cb4952a','b5cb479f7d'
theta_list = np.round(np.arange(0,1.1,0.1),1)

sku_ID_list = ['3c79df1d80','7e4cb4952a','b5cb479f7d']
run_mmnl = True
run_logistic = False
option = ''

for sku_ID in sku_ID_list:
    for theta in [0.8]:
        # read real data
        df1 = pd.read_csv('./../MSOM_data_cleaned/%s%s_%s_V'%(option,sku_ID,str(theta)[:3].replace('.', 'dot')))
        V = df1.values
        
        df2 = pd.read_csv('./../MSOM_data_cleaned/%s%s_%s_choice_ct'%(option,sku_ID,str(theta)[:3].replace('.', 'dot')))
        choice_ct_o = df2.values
        
        sub_sampling = 10000 #10000
        # process readed data, prepare for estimation
        choice_ct_temp = choice_ct_o[:,1]
        V_alltime = np.zeros((len(V), 2, 4))
        V_alltime[:,0,:] = 0 # no-purchase feature
        V_alltime[:,1,:] = V
        choice_ct = np.zeros((len(choice_ct_temp), 2))
        # two "products" - purchase and no-purchase
        choice_ct[:,0] = np.maximum(1 - choice_ct_temp.ravel(),0)
        choice_ct[:,1] = choice_ct_temp.ravel()
        #choice_ct[:,1] = (choice_ct_temp.ravel()>0).astype(int)
        
        #####################################
        
        if run_mmnl == True:
            if len(choice_ct) > sub_sampling:
                V_alltime, choice_ct = sub_sample(V_alltime, choice_ct, sub_sampling)
            
            f, B, alpha, L_rec, b_sol_rec = CGM(V_alltime, choice_ct, 50)
            
            pd.DataFrame(B).to_csv('./../MSOM_data_estimated/freq_%s_%s_%s_B'%(sku_ID,str(theta)[:3].replace('.', 'dot'),str(sub_sampling)), index = False)
            pd.DataFrame(alpha).to_csv('./../MSOM_data_estimated/freq_%s_%s_%s_alpha'%(sku_ID,str(theta)[:3].replace('.', 'dot'), str(sub_sampling)), index = False)    
        
        #------------------------------------------
        #------reduce to binary problem
        mask = (choice_ct[:,1]>1)
        choice_ct[:,1][mask] = 1
        
        if run_logistic == True:
            # reformat data for logistic regression
            X = V_alltime[:,1,:]
            y = choice_ct[:,1]
            clf = LogisticRegression(random_state=0, solver = 'lbfgs',fit_intercept = False).fit(X,y)
            
                