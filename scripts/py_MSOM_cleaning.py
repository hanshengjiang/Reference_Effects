#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 14:45:45 2020

@author: hanshengjiang
"""

'''
MSOM data cleaning code

'''

import numpy as np
import pandas as pd
import datetime as dt
import csv


def search_sku(orders,num_order,num_user):
    '''
    Find the sku with more than num_user of frequent buyers
    Frequent buyers are defined as user who bought one product more than once
    
    '''
    
    df = pd.DataFrame(columns = ['sku_ID'])
    #csv writer
         
    # searching
    for sku_ID in orders['sku_ID'].value_counts().index:
        
        # find orders of only one product/sku
        orders_1p = orders.loc[orders['sku_ID'] == sku_ID]

        # array 
        # number of orders from every user 
        a = orders_1p['user_ID'].value_counts().values

        num_frequent_user = np.sum(a > num_order)

        if num_frequent_user > num_user:
            df = df.append(pd.DataFrame([[sku_ID]], columns = ['sku_ID']))
    df.to_csv('./../MSOM_data_cleaned/frequent_skus_{0}_{1}.csv'.format(str(num_order), str(num_user)))
    return df


def extract_action(orders, clicks, sku_list):
    '''
    Input:
            orders
            clicks
            sku_list: list
                        one sku, 
                    or a list of skus with the same price and the same brand_ID
    Output: 
                |user_ID | time_stamp | action          | original_unit_price | final_unit_price
                |  string| datetime   | 0 = click, 1 = order| NA                 |
          
    '''
    # column names of dataframe
    head = ['user_ID','sku_ID', 'time', 'quantity', 'original_unit_price', 'final_unit_price']
    user_action = pd.DataFrame(columns = head)
    for sku_ID in sku_list:
        
        # order data
        orders_sku = orders.loc[orders['sku_ID'] == sku_ID]
        new_order_row = orders_sku[['user_ID','sku_ID','order_time', 'quantity',\
                                   'original_unit_price','final_unit_price']]
        new_order_row.rename(columns={'order_time':'time'}, inplace=True)
        
        
        # df.append is not in-place
        user_action = user_action.append(new_order_row, ignore_index = True)
        
        # click data
        clicks_sku = clicks.loc[clicks['sku_ID'] == sku_ID]
        new_click_row = pd.DataFrame(columns = head)
        new_click_row[['user_ID','time']] = clicks_sku[['user_ID','request_time']]
        new_click_row.rename(columns={'request_time':'time'}, inplace=True)
        new_click_row['sku_ID'] = sku_ID
        new_click_row['quantity'] = 0
        new_click_row['original_unit_price'] = np.nan
        new_click_row['final_unit_price'] = np.nan 
        
        # df.append is not in-place
        user_action = user_action.append(new_click_row, ignore_index = True)
    return user_action

def clean_action(user_action):
    '''
    Input: 
    user_action
     |user_ID | time_stamp | action              | original_unit_price | final_unit_price
     |  string| datetime   | 0 = click, 1 = order|                     |
     
    
    Output: 
    
    DataFrame: cleaned_action
    '''
    # user with ID '-' are removed
    # it's believed '-' users do not place orders
    cleaned_action = user_action.loc[user_action['user_ID'] != '-'].copy()
    
    
    # delete clicks that appear within the same hour
    # create new column
    cleaned_action['time_hour'] = cleaned_action.apply(lambda row: row['time'].replace(minute=0, second=0), axis=1)
    
    # drop duplicate rows except the first appearance
    cleaned_action = cleaned_action.drop_duplicates(subset=['user_ID', 'quantity','time_hour'], keep='first')
    
    # sort values according to time 
    cleaned_action = cleaned_action.sort_values(by = 'time')
    # fill in nan value in prices, default interpolate method = linear or nearest
    temp_o_price = pd.DataFrame(cleaned_action['original_unit_price']).interpolate()
    cleaned_action['original_unit_price'] = temp_o_price
    temp_f_price = pd.DataFrame(cleaned_action['final_unit_price']).interpolate()
    cleaned_action['final_unit_price'] = temp_f_price
    # further remove nan from prices
    cleaned_action = cleaned_action.loc[cleaned_action['original_unit_price'].notna()]
    cleaned_action = cleaned_action.loc[cleaned_action['final_unit_price'].notna()]
    return cleaned_action

def extract_mmnl_features(cleaned_action, theta):
    '''
    Input: cleaned action, theta
    
    Output: V, choice_ct acoording to given memory parameter
    
    '''
    V = np.zeros((1,4))
    choice_ct = np.zeros((1,1))
    users = pd.unique(cleaned_action['user_ID'])
    for user in users:
        cleaned_action_oneuser = cleaned_action.loc[cleaned_action['user_ID'] == user]

        # make sure rows are sorted in the order of time
        cleaned_action_oneuser = cleaned_action_oneuser.sort_values(by = 'time')

        # initialize reference price as the first original unit price
        # reference_price = cleaned_action_oneuser['original_unit_price'].values[0]
        
        # initialize reference price 
        # as the average of the first original unit price and final unit price
        reference_price = theta * cleaned_action_oneuser['original_unit_price'].values[0]\
                            + (1-theta) * cleaned_action_oneuser['final_unit_price'].values[0]

        for index, row in cleaned_action_oneuser.iterrows():
            # update reference price
            price = row['final_unit_price']

            choice_ct = np.vstack((choice_ct, [[row['quantity']]]))
            V_temp = np.zeros((1,4))
            V_temp[0,0] = 1
            V_temp[0,1] = price
            V_temp[0,2] = max(reference_price - price, 0)
            V_temp[0,3] = min(reference_price - price, 0)
            V = np.vstack((V,V_temp))

            # >0 only update reference price when order
            # >-1 update reference price when click or order
            if row['quantity'] > -1:
                reference_price = theta * reference_price + (1-theta) * price
    return V, choice_ct

def add_reference_price(cleaned_action, user, theta):
    '''
    add a reference price column to the dataframe for one user
    Input: cleaned_action: dataframe
            user: string
    Output: cleaned_action_oneuser
    
    '''
    cleaned_action_oneuser = cleaned_action.loc[cleaned_action['user_ID'] == user]

    # make sure rows are sorted in the order of time
    cleaned_action_oneuser = cleaned_action_oneuser.sort_values(by = 'time')
    
    
    # initialize reference price as the first original unit price
    # reference_price = cleaned_action_oneuser['original_unit_price'].values[0]
    
    # initialize reference price 
    # as the average of the first original unit price and final unit price
    cleaned_action_oneuser['reference_price'] = theta * cleaned_action_oneuser['original_unit_price'].values[0]\
                        + (1-theta) * cleaned_action_oneuser['final_unit_price'].values[0]

    for index, row in cleaned_action_oneuser.iterrows():
        # >0 only update reference price when order
        # >-1 update reference price when click or order
        if row['quantity'] > -1:
            cleaned_action_oneuser.at[index, 'reference_price'] = \
            theta * row['reference_price'] + (1-theta) * row['final_unit_price']
    return cleaned_action_oneuser

from itertools import repeat
import multiprocessing 
from multiprocessing import Pool
         
def extract_mmnl_daily_features(cleaned_action, theta):
    '''
    Input: cleaned action, theta
    
    Output: joined_df acoording to given memory parameter
    
    NOTE: similar to extract_mmnl_features except there is a date of each data
    
    '''
    V = np.zeros((1,4))
    choice_ct = np.zeros((1,1))
    users = pd.unique(cleaned_action['user_ID'])
    
    with Pool(10) as pool:
        results = pool.starmap(add_reference_price, zip(repeat(cleaned_action), users, repeat(theta)))
        
    # join dataframes with the same headers
    joined_df = pd.concat(results)
    
    joined_df['day'] = joined_df.apply(lambda row: float(row['time'].strftime("%d")),axis = 1)
    
    return joined_df



def extract_linear_features(cleaned_action, theta):
    '''
    Input:
    cleaned_action
    |user_ID | time | quantity           | original_unit_price | final_unit_price
    |  string| datetime   | 0 = click, 1 = order|                     |
    
    theta memory parameter
    
    Output:
    features for linea model
    V
    D
    '''
    # number of days in March
    M = 31
    
    V_lin = np.zeros((M,4))
    V_lin[:,0] = 1
    D = np.zeros(M)
    # initialize reference price
    reference_price = 0
    price = 0
    
    for i in range(M):
        df = cleaned_action.loc[(cleaned_action['time'] >= dt.datetime(2018,3,i+1,0,0,0)) &\
                        (cleaned_action['time'] <= dt.datetime(2018,3,i+1,23,59,59))]
        
        # price equal to daily average
        price = df['final_unit_price'].mean()
        V_lin[i,1] = price
        # demand
        D[i] = df['quantity'].sum()
        if i == 0:
            reference_price = df['original_unit_price'].mean()
            V_lin[i,2] = max(reference_price - price, 0)
            V_lin[i,3] = min(reference_price - price, 0)
        else:
            V_lin[i,2] = max(reference_price - price, 0)
            V_lin[i,3] = min(reference_price - price, 0)
            
            # update reference price
            reference_price = theta * reference_price + (1-theta) * price
    return V_lin,D
