#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:10:48 2020

@author: hanshengjiang
"""
from py_estimation import *
from py_MSOM_cleaning import *

# 'skus' table
skus = pd.read_csv('./../MSOM_Data/JD_sku_data.csv')

# 'users' table
users = pd.read_csv('./../MSOM_Data/JD_user_data.csv')

# 'clicks' table
clicks = pd.read_csv('./../MSOM_Data/JD_click_data.csv')
clicks['request_time'] = pd.to_datetime(clicks['request_time']) #string to datetime 

# 'orders' table
orders = pd.read_csv('./../MSOM_Data/JD_order_data.csv')
# to datetime 
orders['order_time'] = pd.to_datetime(orders['order_time'])
orders['order_date'] = pd.to_datetime(orders['order_date'])
# add brand_ID
orders = pd.merge(orders, skus[['sku_ID','brand_ID']], on='sku_ID')
# move brand_ID to the front
cols = orders.columns.tolist()
cols = cols[-1:] + cols[1:]
orders = orders[cols]

# 'delivery' table
# delivery = pd.read_csv('./MSOM_data/JD_delivery_data.csv')

# 'inventory' table
# inventory = pd.read_csv('./MSOM_data/JD_inventory_data.csv')

# 'network' table
# network = pd.read_csv('./MSOM_data/JD_network_data.csv')


#################
# Selected SKU_ID that are 'frequently' bounght
# csv file in ./MSOM_data_cleaned/archive/frequent_skus_1_100.csv
# code for generating the csv file can be found in ./MSOM_data_cleaning.ipynb
#################

sku_ID_list = ['3c79df1d80','7e4cb4952a','b5cb479f7d','8dc4a01dec','adfedb6893']

for sku_ID in sku_ID_list:
    user_action = extract_action(orders, clicks, [sku_ID])
    user_action.to_csv('./../MSOM_data_cleaned/user_action_%s.csv'%sku_ID, index = False)
    cleaned_action = clean_action(user_action)
    cleaned_action.to_csv('./../MSOM_data_cleaned/cleaned_action_%s.csv'%sku_ID, index = False)
