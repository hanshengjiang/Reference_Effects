#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 10:53:15 2020

@author: hanshengjiang
"""

'''

linear estimation 

'''

from py_estimation import *
from py_MSOM_cleaning import *
from sklearn import linear_model


# set sku_ID
# '3c79df1d80',
sku_ID_list = ['adfedb6893', '3c79df1d80', '7e4cb4952a', 'b5cb479f7d']

option = ''
for sku_ID in sku_ID_list:
    
    # chosen according to cross-validation
    for theta in [0.8]:
        
            
        # read real data
        df = pd.read_csv('./../MSOM_data_cleaned/%s%s_%s_V_lin'%(option,sku_ID,str(theta).replace('.', 'dot')))
        V_lin = df.values
        df = pd.read_csv('./../MSOM_data_cleaned/%s%s_%s_D'%(option,sku_ID,str(theta).replace('.', 'dot')))
        D = df.values
        
        
        # do NOT add extra constant intercept 
        reg = linear_model.LinearRegression(fit_intercept = False)

        train_end = 31
        V_lin_train = V_lin[:train_end,:]
        D_train = D[:train_end,:]
        
        reg.fit(V_lin_train, D_train)
        
        pd.DataFrame(reg.coef_).to_csv('./../MSOM_data_estimated/{}{}_{}_linear_coef_{}'\
                    .format(option,sku_ID,str(theta).replace('.', 'dot'),train_end), index = False)
        
        



