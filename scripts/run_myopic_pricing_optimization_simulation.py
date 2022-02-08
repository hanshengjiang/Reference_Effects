#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: hanshengjiang
"""
from optimal_pricing_policy_exp_update import *
import random
import time

import sys

from itertools import repeat
import multiprocessing 
from multiprocessing import Pool

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please provide experiments configuration!")
    
    config = sys.argv[1]
    
    # default
    if len(sys.argv) < 7:
        epsilon = 0.01 # price discretization accuracy
        L = 0 #lower bound of price
        H = 10 #upper bound of price
        T = np.inf
        theta = 0.8
        gamma = 0.95
    # otherwise take arguments from command line
    else:
        #sys_argv[0] is the name of the .py file
        epsilon = float(sys.argv[2]) 
        L = float(sys.argv[3]) # number of data points
        H = float(sys.argv[4])
        T = float(sys.argv[5])
        if T != np.inf:
            T = int(T)
        theta = float(sys.argv[6]) 
        
    if config == 'myopic_1':
        coefficients = (1,1,1,1)
        weights = (1,0,0)
        demand_name = 'mmnl'
            
    elif config == 'myopic_2':
        coefficients = (2,2,0.2,0.2)
        weights = (1,0,0)
        demand_name = 'mmnl'        

price_list = np.arange(L-epsilon,H+2*epsilon,epsilon)


def compute_myopic(c):    
    coefficients = (2, 2,c,2)
    #---------------------------------#
    # myopic pricing policy
    # name prefix for saved plots
    coeffname = 'myopic_inf_hor_fixed_price_range'+str(L).replace(".","dot")+'_'+str(H).replace(".","dot")+'_'+str(T)+'_'\
    +str(coefficients).replace(" ", "_").replace(".","dot")\
    +str(weights).replace(" ","_").replace(".","dot")+'_'+str(theta).replace(".", "dot")+'_'+str(gamma).replace(".", "dot")+demand_name
    
    # optimal pricing policy
    V,mu \
    = inf_hor_pricing_pricerange(L,H,theta,epsilon,np.inf,gamma,coefficients,weights,demand_name)
    
    pd.DataFrame(V).to_csv('./../pricing_output/%s_V'%coeffname, index = False, header = False)
    pd.DataFrame(mu).to_csv('./../pricing_output/%s_mu'%coeffname, index = False, header = False)
    
    # myopic pricing policy
    V_m,mu_m \
     = myopic_inf_hor_pricing_pricerange(L,H,theta,epsilon,T,gamma,coefficients,weights,demand_name)
    
    pd.DataFrame(V_m).to_csv('./../pricing_output/%s_V_m'%coeffname, index = False, header = False)
    pd.DataFrame(mu_m).to_csv('./../pricing_output/%s_mu_m'%coeffname, index = False, header = False)
    
    #---------------------------------#
    
    '''
    #---------------price path-------------------------#
    # plot price path
    start_ = 0 # staring time for plots
    end_ = 20
    # i_li = random.sample(range(len(V)),1)
    i_li = []
    i_li.append(int((H/epsilon)/2)+1)
    for i in i_li:
        
        c = []
        id = i
        # need to add the first reference price
        c.append(id)
        #apply the policy a few times
        for j in range(end_):
            c.append(int(mu[id]))
            id = int(mu[id])
        
        c_m = []
        id_m = i
        # need to add the first reference price
        c_m.append(id_m)
        #apply the policy a few times
        for j in range(end_):
            c_m.append(int(mu_m[id_m]))
            id_m = int(mu_m[id_m])
        
        fig3, ax3 = plt.subplots()
        plt.tight_layout()
        price_path = (price_list[c[1:]] - theta * price_list[c[:-1]])/(1-theta)
        ax3.plot(np.arange(start_,end_),price_path[start_:],\
                 marker = 'o',mfc = 'none',color = 'black',linestyle = '--')
        
        price_path_m = (price_list[c_m[1:]] - theta * price_list[c_m[:-1]])/(1-theta)
        ax3.plot(np.arange(start_,end_),price_path_m[start_:],\
                 marker = 'o',mfc = 'none', color = 'black',linestyle = '--')
        
        
        ax3.set_ylim(L,H+1)
        ax3.set_xticks(np.arange(start_,end_+1,5))
        ax3.set_xlabel('Time', size = 16)
        #ax3.set_ylabel('optimal price path',size = 16)
        ax3.set_ylabel('Price',size = 16)
        coeffname_r0 = coeffname + 'r0=' + str(round(price_list[i],2)).replace(".","dot")
        fig3.savefig('./../pricing_output/%s_price_path'%coeffname_r0, dpi= 300)
    '''
    
    print(c, (np.divide(V - V_m,V)).sum()/len(V))
    
with Pool(9) as pool:
    pool.map(compute_myopic, np.arange(1,5,0.5))
    
 
 
 