#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:35:59 2020

@author: hanshengjiang
"""
from py_estimation import *
from py_MSOM_cleaning import *

# set sku_ID.

# set theta
# theta_list = np.round(np.arange(0.1,1,0.1),1) 

# '3c79df1d80',
sku_ID_list = ['3c79df1d80', '7e4cb4952a', 'b5cb479f7d', '8dc4a01dec', 'adfedb6893']

# ['adfedb6893', '3c79df1d80', 'b5cb479f7d']:

from itertools import repeat
import multiprocessing 
from multiprocessing import Pool


    
def estimate_JD(sku_ID, rd_seed):
    np.random.seed(rd_seed)
    
    # chosen according to cross-validation
    for theta in [0.8]:
    
        # sample size, total number of rows is too large
        sub_sampling = 2000
        
        # cleaned_action = pd.read_csv('./MSOM_data_cleaned/cleaned_action_%s.csv'%sku_ID, parse_dates = ['time'])
        
        #---------------------------------------#
            
        # read real data
        df = pd.read_csv('./../MSOM_data_cleaned/%s_%s_V'%(sku_ID,str(theta)[:3].replace('.', 'dot')))
        V = df.values
        
        df = pd.read_csv('./../MSOM_data_cleaned/%s_%s_choice_ct'%(sku_ID,str(theta)[:3].replace('.', 'dot')))
        choice_ct_temp = df.values
        # process readed data, prepare for estimation
        V_alltime = np.zeros((len(V), 2, 4))
        V_alltime[:,0,:] = 0 # no purchase feature
        V_alltime[:,1,:] = V
        
        # change sign of features (1, -p, (r-p)_+, (r-p)_-)
        V_alltime[:,1,1] = - V_alltime[:,1,1]
        
        choice_ct = np.zeros((len(choice_ct_temp), 2))
        # two "products" - purchase and no-purchase
        choice_ct[:,0] = np.maximum(1 - choice_ct_temp.ravel(),0)
        choice_ct[:,1] = np.minimum(1, choice_ct_temp.ravel())
        #choice_ct[:,1] = (choice_ct_temp.ravel()>0).astype(int)
        #---------------------------------------#
        
        if len(choice_ct) > sub_sampling:
            V_alltime, choice_ct = sub_sample(V_alltime, choice_ct, sub_sampling)
        
        
        
        start_time = time.time()
        f, B, vB, alpha, L_rec = CGM(V_alltime, choice_ct, 60)
        print("--- %s seconds ---" % (time.time() - start_time))
        
        pd.DataFrame(B).to_csv('./../MSOM_data_estimated/%s_%s_%s_B'%(sku_ID,str(theta).replace('.', 'dot'),str(sub_sampling)), index = False, header = False)
        pd.DataFrame(vB).to_csv('./../MSOM_data_estimated/%s_%s_%s_vB'%(sku_ID,str(theta).replace('.', 'dot'),str(sub_sampling)), index = False, header = False)
        pd.DataFrame(alpha).to_csv('./../MSOM_data_estimated/%s_%s_%s_alpha'%(sku_ID,str(theta).replace('.', 'dot'), str(sub_sampling)), index = False, header = False)
        
        
        # read file
        # note: header = None is important
        B = pd.read_csv('./../MSOM_data_estimated/%s_%s_%s_B'%(sku_ID,str(theta).replace('.', 'dot'),str(sub_sampling)), header = None).values
        vB = pd.read_csv('./../MSOM_data_estimated/%s_%s_%s_vB'%(sku_ID,str(theta).replace('.', 'dot'),str(sub_sampling)), header = None).values
        alpha = pd.read_csv('./../MSOM_data_estimated/%s_%s_%s_alpha'%(sku_ID,str(theta).replace('.', 'dot'), str(sub_sampling)), header = None).values
        
        alpha = alpha.ravel()
        
        alphaBvB = np.append(np.reshape(alpha, (len(alpha),1)),\
                             (np.append(B,vB, axis = 0)).T, axis = 1)
        idx = (-alpha).argsort()
        alphaBvB =  alphaBvB[idx]
        
        pd.DataFrame(alphaBvB).to_csv('./../MSOM_data_estimated/%s_%s_%s_alphaBvB.csv'%(sku_ID,str(theta).replace('.', 'dot'), \
                                      str(sub_sampling)), index = False, header = False, float_format='%.3f')
        
        
with Pool(3) as mmnl_pool:
    mmnl_pool.starmap(estimate_JD, [('adfedb6893',1), ('3c79df1d80',2), ('b5cb479f7d',3)])