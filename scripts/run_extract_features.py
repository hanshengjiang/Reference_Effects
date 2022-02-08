#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:18:09 2020

@author: hanshengjiang
"""
from py_estimation import *
from py_MSOM_cleaning import *

# from run_data_cleaning import *
# set theta
theta_list = np.round(np.arange(0,1,0.1),1) 

# '3c79df1d80','7e4cb4952a','b5cb479f7d','8dc4a01dec',
# set sku_ID
#sku_ID_list = ['adfedb6893']
sku_ID_list = ['adfedb6893', '3c79df1d80', 'b5cb479f7d']
# Whether or not to extract mmnl or linear features
mmnl = False
linear = True
mmnl_daily = False

for sku_ID in sku_ID_list:
    # read cleaned data
    cleaned_action = pd.read_csv('./../MSOM_data_cleaned/cleaned_action_%s.csv'%sku_ID, parse_dates = ['time'])
       
    if mmnl == True:
        for theta in [0.8]:
            
            V, choice_ct = extract_mmnl_features(cleaned_action,theta)
            
            # save to csv
            pd.DataFrame(V).to_csv('./../MSOM_data_cleaned/%s_%s_V'\
                                   %(sku_ID,str(theta).replace('.', 'dot')), index = False)
            # save to csv
            pd.DataFrame(choice_ct).to_csv('./../MSOM_data_cleaned/%s_%s_choice_ct'\
                                           %(sku_ID,str(theta).replace('.', 'dot')), index = False)
    if mmnl_daily == True:
        for theta in [0.8]:
            
            df = extract_mmnl_daily_features(cleaned_action, theta)
            
            V_daily = np.zeros((df.values.shape[0],6))
            V_daily[:,0] =  df['day'].values
            V_daily[:,1] = 1
            V_daily[:,2] = df['final_unit_price'].values
            V_daily[:,3] = np.maximum(df['reference_price'].values - df['final_unit_price'].values, 0)
            V_daily[:,4] = np.minimum(df['reference_price'].values - df['final_unit_price'].values, 0)
            V_daily[:,5] = df['quantity'].values
            
            # save to csv
            pd.DataFrame(V_daily).to_csv('./../MSOM_data_cleaned/%s_%s_V_daily'\
                                   %(sku_ID,str(theta).replace('.', 'dot')), index = False)
            
    if linear == True:
        for theta in theta_list:
            
            V_lin, D = extract_linear_features(cleaned_action, theta)
            
            # save to csv
            pd.DataFrame(V_lin).to_csv('./../MSOM_data_cleaned/%s_%s_V_lin'\
                                   %(sku_ID,str(theta).replace('.', 'dot')), index = False)
            # save to csv
            pd.DataFrame(D).to_csv('./../MSOM_data_cleaned/%s_%s_D'\
                                           %(sku_ID,str(theta).replace('.', 'dot')), index = False)
        