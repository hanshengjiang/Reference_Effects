#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:18:47 2020

@author: hanshengjiang

"""
'''
Extract features from data
'''

from run_data_cleaning import *

from py_estimation import *
from py_MSOM_cleaning import *

# '3c79df1d80', '7e4cb4952a', 'b5cb479f7d', '8dc4a01dec',
sku_ID_list = [ 'adfedb6893']
#theta_list = np.round(np.arange(0,1.1,0.1),1)
theta_list = [0.0]
# sku_ID = '7e4cb4952a'

for sku_ID in sku_ID_list:
    user_orders = orders.loc[orders['sku_ID']== sku_ID ]['user_ID'].value_counts()
    arr = pd.DataFrame(user_orders)
    arr = arr.rename(columns = {'user_ID':'counts'})
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # frequent user of this one product
    freq_user_list = list(arr.loc[arr['counts']>1].index)
    #++++++++++++++++++++++++++++++++++++++++++++++++#
    
    orders_freq = orders.loc[orders['user_ID'].isin(freq_user_list)]
    clicks_freq = clicks.loc[clicks['user_ID'].isin(freq_user_list)]
    
    user_action = extract_action(orders_freq, clicks_freq, ['b5cb479f7d'])
    cleaned_action = clean_action(user_action)
    cleaned_action.to_csv('./../MSOM_data_cleaned/freq_cleaned_action_%s.csv'%sku_ID)
    
    mmnl = False
    linear = True
    # set theta
    #theta_list = np.round(np.arange(0.0,1,0.1),1) 
    
    # read cleaned data
    # cleaned_action = pd.read_csv('./MSOM_data_cleaned/freq_cleaned_action_%s.csv'%sku_ID, parse_dates = ['time'])
       
    if mmnl == True:
        for theta in theta_list:
            
            V, choice_ct_o = extract_mmnl_features(cleaned_action,theta)
            
            # save to csv
            pd.DataFrame(V).to_csv('./../MSOM_data_cleaned/freq_%s_%s_V'\
                                   %(sku_ID,str(theta).replace('.', 'dot')), index = False)
            # save to csv
            pd.DataFrame(choice_ct).to_csv('./../MSOM_data_cleaned/freq_%s_%s_choice_ct'\
                                           %(sku_ID,str(theta).replace('.', 'dot')), index = False)
            
    if linear == True:
        for theta in theta_list:
            
            V_lin, D = extract_linear_features(cleaned_action, theta)
            
            # save to csv
            pd.DataFrame(V_lin).to_csv('./../MSOM_data_cleaned/freq_%s_%s_V_lin'\
                                   %(sku_ID,str(theta).replace('.', 'dot')), index = False)
            # save to csv
            pd.DataFrame(D).to_csv('./../MSOM_data_cleaned/freq_%s_%s_D'\
                                           %(sku_ID,str(theta).replace('.', 'dot')), index = False)

    