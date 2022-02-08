#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 21:00:52 2020

@author: hanshengjiang
"""

from optimal_pricing_policy_exp_update import *

# change plot fonts
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix",
     "font.size": 16}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
# +++++++++++++++++++++++++++++++++++++++++++++

import random
np.random.seed(626)
random.seed(626)

####Ningyuan's coefficients for linear
# demand_name = 'linear'
# for linear demand, all zero means fake segment
# coefficients = (1,0.3,0,0.1,4,2,1.75,2)
# coefficients = (0,0,0,0,4,2,1.75,1.75)


# +++++++++++++++++++++++++++++++++
# Read Estimated Results
# +++++++++++++++++++++++++++++++++

# sku_ID_list = ['adfedb6893','3c79df1d80', '7e4cb4952a', 'b5cb479f7d']

# '8dc4a01dec' 
# ('adfedb6893',50,160,10000)\
# ('3c79df1d80',30,60,13000),('7e4cb4952a', 20, 50,13000)
# ('3c79df1d80',30,60,10000)
option = ''

for theta in [0.8]:
    for (sku_ID, L,H, sub_sampling) in [ ('adfedb6893',50,160,2000),('3c79df1d80',30,60,2000),('b5cb479f7d', 20,50,2000)]:
        
        # timestamp = datetime.today().strftime('%m_%d_%Y')
        timestamp = "01_16_2022"
        
        
        # L and H should be adjusted with sku_ID
        epsilon = (H-L)/500 
#        L = 60 # lowest price changes with sku
#        H = 150 # highest price changes with sku
        
        
        T = np.inf
        gamma = 0.95 # discounted factor
        price_list = np.arange(L-epsilon,H+ 2*epsilon,epsilon)
        
    
    
        # estimation results produced by "run_mmnl_estimation_simulation.py"
        B = pd.read_csv('./../MSOM_data_estimated/{}/CGM/B_{}.csv'.format(sku_ID,timestamp), header = None).values
        vB = pd.read_csv('./../MSOM_data_estimated/{}/CGM/vB_{}.csv'.format(sku_ID,timestamp), header = None).values
        alpha = pd.read_csv('./../MSOM_data_estimated/{}/CGM/alpha_{}.csv'.format(sku_ID,timestamp), header = None).values
        
        # processing
        alpha = alpha.ravel()
        BvB = np.append(B,vB, axis = 0)
        coefficients = (BvB.T).ravel()
        weights = alpha
        demand_name = 'mmnl_ext'
        
        
        #optimal pricing policy
        V,mu \
        = inf_hor_pricing_pricerange(L,H,theta,epsilon,np.inf,gamma,coefficients,weights,demand_name)
        
        pd.DataFrame(V).to_csv('./../MSOM_data_estimated/{}{}_{}_{}_V'.format(option,sku_ID,str(theta)[:3].replace('.', 'dot'),str(sub_sampling)))
        pd.DataFrame(mu).to_csv('./../MSOM_data_estimated/{}{}_{}_{}_mu'.format(option,sku_ID,str(theta)[:3].replace('.', 'dot'),str(sub_sampling)))
        
        
        #+++++++++++++++++++++++++++++++++
        # plot results
        #+++++++++++++++++++++++++++++++
        
        
        # name prefix for saved plots
        coeffname = 'pricing_opt'+ '_'+demand_name+'_'\
        +sku_ID+'_'+ str(theta)[:3].replace('.', 'dot') +'_' + str(sub_sampling)+'_'\
        +str(L).replace(".","dot")+'_'+str(H).replace(".","dot")+'_'+str(T)+'_' \
        + str(epsilon).replace(".","dot")+'_'+ str(gamma)[:4].replace('.', 'dot')
        
#        # reference price
#        fig, ax = plt.subplots()
#        ax.plot(price_list[np.arange(len(mu)).astype(int)], price_list[mu.astype(int)],\
#                color = 'black',linestyle = '--',marker = 'o',mfc = 'none');
#        ax.set_xlabel(r"$r_t$",size = 30);
#        ax.set_ylabel(r"$r_{t+1}$",size = 30);
#        ax.set_ylim(L-1,H+1)
#        fig.savefig('./pricing_output/%s_referenceprice'%coeffname, dpi= 300)
#        
#        # value function as a function of reference price
#        fig1, ax1 = plt.subplots()
#        ax1.plot(price_list[np.arange(len(mu)).astype(int)], \
#                 V, color = 'black',mfc = 'none');
#        ax1.set_xlabel(r"$r_t$",size = 30);
#        ax1.set_ylabel(r"$V^*(r_t)$",size = 30);
#        #ax2.set_ylim(,)
#        fig1.savefig('./pricing_output/%s_value_function'%coeffname, dpi= 300)
#        
#        # pricing policy
#        fig2, ax2 = plt.subplots()
#        ax2.plot(price_list[np.arange(len(mu)).astype(int)], 
#                 (price_list[mu.astype(int)] - theta * price_list[np.arange(len(mu)).astype(int)])/(1-theta),\
#                 color = 'black',linestyle = '--',marker = 'o',mfc = 'none');
#        ax2.set_xlabel(r"$r_t$",size = 30);
#        ax2.set_ylabel(r"$p^*_t$",size = 30);
#        ax2.set_ylim(L-1,H+1)
#        fig2.savefig('./pricing_output/%s_price'%coeffname, dpi= 300)
        
        # plot price path
        i_li =  random.sample(range(len(V)),10)
        i_li.append(int((H/epsilon)/2)+1)
        for i in i_li:
            c = []
            # choose different initial reference price
            id = i
            
            # need to add the first reference price
            c.append(id)

            #apply the policy a few times
            for j in range(30):
                c.append(int(mu[id]))
                id = int(mu[id])
            
            fig3, ax3 = plt.subplots()
            plt.tight_layout()
            price_path = (price_list[c[1:]] - theta * price_list[c[:-1]])/(1-theta)
            ax3.plot(range(len(price_path)),price_path,color = 'black',mfc = 'none',marker = 'o',linestyle = '--')
            ax3.set_ylim(L-1,H+1)
            ax3.set_xlabel('Time')
            #ax3.set_ylabel('optimal price path')
            ax3.set_ylabel('Price')
            coeffname_r0 = coeffname + '_r0=' + str(round(price_list[id],2)).replace(".","dot")
            fig3.savefig('./../pricing_output/%s_price_path'%coeffname_r0, dpi= 300,bbox_inches='tight')
