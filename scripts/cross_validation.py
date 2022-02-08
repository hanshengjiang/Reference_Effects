#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:15:11 2020

@author: hanshengjiang
"""

###################################
# Run Cross validation to choose theta (memory parameter)
# based on MMNL model
####################################

from py_estimation import *
from py_MSOM_cleaning import *
import pandas as pd
import random
import os
from sklearn.model_selection import KFold
import time
import sys



def cross_validation_mmnl_score(sku_ID,k,theta,sub_sampling):
    '''
    k-fold cross validation
    
    sub_sampling = False or an integer
    
    return the score
    '''
    
    # acquire data from scratch if not available
    #---------------------------------------------#
    if not (os.path.isfile('./../MSOM_data_cleaned/%s_%s_V'%(sku_ID,str(theta).replace('.', 'dot'))) and 
            os.path.isfile('./../MSOM_data_cleaned/%s_%s_choice_ct'%(sku_ID,str(theta).replace('.', 'dot'))) ):
        # if file does not exist
        
        print('################\n start data cleaning\n ################')
        # do data clean and generate feature filess
        if not os.path.isfile('./../MSOM_data_cleaned/user_action_%s'%sku_ID):
            user_action = extract_action(orders, clicks, [sku_ID])
        else:
            user_action = pd.read_csv('./../MSOM_data_cleaned/user_action_%s'%sku_ID)
        cleaned_action = clean_action(user_action)
        
        
        V, choice_ct = extract_mmnl_features(cleaned_action, theta)
        
        # save to csv
        pd.DataFrame(V).to_csv('./../MSOM_data_cleaned/%s_%s_V'\
                    %(sku_ID,str(theta).replace('.', 'dot')), index = False)
        # save to csv
        pd.DataFrame(choice_ct).to_csv('./../MSOM_data_cleaned/%s_%s_choice_ct'\
            %(sku_ID,str(theta).replace('.', 'dot')), index = False)
    ###########################################################################
    
    # read real data
    df = pd.read_csv('./../MSOM_data_cleaned/%s_%s_V'%(sku_ID,str(theta).replace('.', 'dot')))
    V = df.values
    df = pd.read_csv('./../MSOM_data_cleaned/%s_%s_choice_ct'%(sku_ID,str(theta).replace('.', 'dot')))
    choice_ct_temp = df.values
    
    # process readed data, prepare for estimation
    V_alltime = np.zeros((len(V), 2, 4))
    V_alltime[:,0,:] = 0 # no purchase feature
    V_alltime[:,1,:] = V
    choice_ct = np.zeros((len(choice_ct_temp), 2))
    # two "products" - purchase and no-purchase
    choice_ct[:,0] = 1 - choice_ct_temp.ravel()
    choice_ct[:,1] = choice_ct_temp.ravel()
    #####################################
    
    
    if sub_sampling != False:
        
        V_alltime, choice_ct = sub_sample(V_alltime, choice_ct, subampling)
    
    # create k fold    
    kf = KFold(n_splits = k, shuffle = True, random_state = 626)
    
    # initialize score
    score = 0.0
    
    for train_index, test_index in kf.split(np.zeros((len(V_alltime),1))):
        f, B, alpha, L_rec, b_sol_rec = CGM(V_alltime[train_index,:,:],choice_ct[train_index,:],50)
        score += neg_log_likelihood( B, alpha,V_alltime[test_index,:,:],choice_ct[test_index,:] )
    score = score/k
    return score


def cross_validation_linear_score(sku_ID,k,theta,sub_sampling):
    '''
    k-fold cross validation
    
    sub_sampling = False or an integer
    
    return the score
    '''
    # acquire data from scratch (by calling data cleaning functions if not available
    ######################################################
    if not (os.path.isfile('./../MSOM_data_cleaned/%s_%s_V_lin'%(sku_ID,str(theta).replace('.', 'dot'))) and 
            os.path.isfile('./../MSOM_data_cleaned/%s_%s_D'%(sku_ID,str(theta).replace('.', 'dot'))) ):
        # if file does not exist
        
        print('################\n start data cleaning\n ################')
        # do data clean and generate feature filess
        if not os.path.isfile('./../MSOM_data_cleaned/user_action_%s'%sku_ID):
            user_action = extract_action(orders, clicks, [sku_ID])
        else:
            user_action = pd.read_csv('./../MSOM_data_cleaned/user_action_%s'%sku_ID, parse_dates= ['time'])
        cleaned_action = clean_action(user_action)
        
      
        V_lin, D = extract_linear_features(cleaned_action, theta)
        
        # save to csv
        pd.DataFrame(V_lin).to_csv('./../MSOM_data_cleaned/%s_%s_V_lin'\
                    %(sku_ID,str(theta).replace('.', 'dot')), index = False)
        # save to csv
        pd.DataFrame(D).to_csv('./../MSOM_data_cleaned/%s_%s_D'\
            %(sku_ID,str(theta).replace('.', 'dot')), index = False)
    ########################################################################
    
    
    # read real data
    df = pd.read_csv('./../MSOM_data_cleaned/%s_%s_V_lin'%(sku_ID,str(theta).replace('.', 'dot')))
    V_lin = df.values
    df = pd.read_csv('./../MSOM_data_cleaned/%s_%s_D'%(sku_ID,str(theta).replace('.', 'dot')))
    D = df.values
    
    # subsampling is not needed for linear models
    ##############################
#    if sub_sampling != False:
#        
#        # get subsample of data
#        random.seed(626)
#        
#        # keep the ratio of purchase vs. no-purchse unchanged
#        purchase_index = np.argwhere(choice_ct[:,0] == 0).ravel()
#        no_purchase_index = np.argwhere(choice_ct[:,0] == 1).ravel()
#        
#        ratio_p = len(purchase_index)/len(choice_ct)
#        num_p = int(sub_sampling * ratio_p)+1
#        num_np = sub_sampling - num_p
#        
#        sample_purchase_index = np.array( random.sample(list(purchase_index), num_p) )
#        sample_no_purchase_index = np.array( random.sample(list(no_purchase_index), num_np) )
#        
#        # combine to get sample index
#        sample_index = np.hstack((sample_purchase_index, sample_no_purchase_index))
#        
#        V_alltime = np.copy(V_alltime[sample_index,:,:])
#        choice_ct = np.copy(choice_ct[sample_index,:])
    ####################################
    
    # create k fold    
    kf = KFold(n_splits = k, shuffle = True, random_state = 626)
    
    # initialize score
    score = 0.0
    
    for train_index, test_index in kf.split(np.zeros((len(V_alltime),1))):
        
        # training
        coef_, error = linear_model(V_lin, D)
        
        # validating
        score += np.linalg.norm(np.matmul(V_lin[test_index,:],coef_) - choice_ct[test_index])\
                                /np.sqrt(len(test_index))
    score = score/k
    return score

def cross_validation_over_theta(sku_ID, k, theta_list,sub_sampling, model_name):
    '''
    perform k-fold cross-validation for a list of theta and 
    save the score to file
    '''
    n_list = len(theta_list)
    CV = np.zeros((n_list, 2))
    CV[:,0] = np.array(theta_list)
    for i in range(n_list):
        start_time = time.time()
        if model_name == 'mmnl':
            CV[i,1] = cross_validation_mmnl_score(sku_ID,k,CV[i,0],sub_sampling)
        elif model_name == 'linear':
            CV[i,1] = cross_validation_linear_score(sku_ID,k,CV[i,0],sub_sampling)
        else:
            sys.exit('Wrong model name!')
        end_time = time.time()
        print('##############################\n')
        print('theta: {:.2f}, score: {:.5f}, time: {:.2f} seconds'.format(CV[i,0], CV[i,1], end_time - start_time ))
        print('\n##############################')
    # save to csv
    pd.DataFrame(CV).to_csv('./../MSOM_data_cleaned/%s_CrossValidation_%s_%s_sample'%(model_name,sku_ID,str(sub_sampling)), index = False)
  
# cross_validation_over_theta('3c79df1d80', 5, np.round(np.arange(0.1,1,0.1),1), 10000, 'mmnl')
cross_validation_over_theta('3c79df1d80', 5, np.round(np.arange(0.1,1,0.1),1), False, 'linear')
