#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 14:22:42 2020

@author: hanshengjiang
"""

'''
use simulated 
estimation by logistic, EM, and nonparametric MLE
plot results
'''


from optimal_pricing_policy_exp_update import *
from mmnl_simulation import *
import random
import pickle

# timestamp = datetime.today().strftime('%m_%d_%Y')
timestamp = "01_10_2022"

T_est = 500
n = 2
d = 4 
folder_id = 1

#----------------------------------------------------
epsilon = 0.01 # price discretization accuracy default 0.01
L = 0 #lower bound of price
H = 10 #upper bound of price
T = np.inf
theta = 0.8 # memory parameter
gamma = 0.95 # discount factor
price_list = np.arange(L-epsilon,H+2*epsilon,epsilon)

#----------------------------
# NEED to be consistent with estimation
BH = 10
#-----------------------------

#---------------------------------------------------------------------------------------
# mmnl (heterogeneous models)
#---------------------------------------------------------------------------------------


# estimation results produced by "run_mmnl_estimation_simulation.py"
B = pd.read_csv('./../simulation_results/CGM/{}/B_{}_{}_{}_{}.csv'.format(folder_id,T_est,n,d,timestamp), header = None).values
vB = pd.read_csv('./../simulation_results/CGM/{}/vB_{}_{}_{}_{}.csv'.format(folder_id,T_est,n,d,timestamp), header = None).values
alpha = pd.read_csv('./../simulation_results/CGM/{}/alpha_{}_{}_{}_{}.csv'.format(folder_id,T_est,n,d,timestamp), header = None).values



alpha = alpha.ravel()

BvB = np.append(B,vB, axis = 0)

coefficients = (BvB.T).ravel()

weights = alpha
demand_name = 'mmnl_ext'

V_1,mu_1 \
= inf_hor_pricing_pricerange(L,H,theta,epsilon,np.inf,gamma,coefficients,weights,demand_name)

#------------------------------------------------

#---------------------------------------------------------------------------------------
# EM 
#---------------------------------------------------------------------------------------


# estimation results produced by "run_mmnl_estimation_simulation.py"
B_EM = pd.read_csv('./../simulation_results/EM/{}/B_EM.csv'.format(folder_id), header = None).values
alpha_EM = pd.read_csv('./../simulation_results/EM/{}/alpha_EM.csv'.format(folder_id), header = None).values


alpha_EM = alpha_EM.ravel()
coefficients = (B_EM.T).ravel()
weights = alpha_EM
demand_name = 'mmnl'

V_2,mu_2 \
= inf_hor_pricing_pricerange(L,H,theta,epsilon,np.inf,gamma,coefficients,weights,demand_name)




#---------------------------------------------------------------------------------------
# logistic (homogeneous)
#---------------------------------------------------------------------------------------

# estimation results produced by "run_mmnl_estimation_simulation.py"
B_logit = pd.read_csv('./../simulation_results/Logit/{}/B_logit_{}_{}_{}_{}.csv'.format(folder_id,T_est,n,d,timestamp), header = None).values
alpha_logit = pd.read_csv('./../simulation_results/Logit/{}/alpha_logit_{}_{}_{}_{}.csv'.format(folder_id,T_est,n,d,timestamp), header = None).values



alpha_logit = alpha_logit.ravel()
coefficients = (B_logit.T).ravel()
weights = alpha_logit
demand_name = 'mmnl'

V_3,mu_3 \
= inf_hor_pricing_pricerange(L,H,theta,epsilon,np.inf,gamma,coefficients,weights,demand_name)

#------------------------------------------------


#---------------------------------------------------------------------------------------
# linear 
#---------------------------------------------------------------------------------------



# estimation results produced by "run_mmnl_estimation_simulation.py"
B_lm = pd.read_csv('./../simulation_results/Linear/{}/B_lm_{}_{}_{}_{}.csv'.format(folder_id,T_est,n,d,timestamp), header = None).values
alpha_lm = pd.read_csv('./../simulation_results/Linear/{}/alpha_lm_{}_{}_{}_{}.csv'.format(folder_id,T_est,n,d,timestamp), header = None).values



alpha_lm = alpha_lm.ravel()
coefficients = (B_lm.T).ravel()
weights = alpha_lm

demand_name = 'linear'
V_4,mu_4 \
= inf_hor_pricing_pricerange(L,H,theta,epsilon,np.inf,gamma,coefficients,weights,demand_name)



#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

label_li = ['Nonparametric MLE', 'Mixed Logit via EM', 'Single Logit', 'Linear']
color_li = ["#0075DC", "#00998F", "#FFA405", "#F0A3FF" ]
linestyle_li = [   'dotted', #'dotted'
                 (0, (5, 5)), #'dashed'
                 (0, (3, 1, 1, 1)), #'dashdotted'
                 'dashdot' 
                ]
marker_li = ["o", "d", ">", "s"]






#--------------------plots----------------------------
coeffname = 'estimate_optimize'+str(L).replace(".","dot")+'_'+str(H).replace(".","dot")\
+'_'+str(theta).replace(".", "dot")+'_'+str(gamma).replace(".", "dot")

#------------------------
# plot price path
#------------------------
start_ = 0 # staring time for plots
end_ = 30
i_li = random.sample(range(len(V_1)),10)
i_li.append(int((H/epsilon)/2)+1)
for i in i_li:
    c_1 = []
    c_2 = []
    c_3 = []
    c_4 = []
    
    id = i
    #apply the policy a few times
    for j in range(end_):
        c_1.append(int(mu_1[id]))
        id = int(mu_1[id])
    price_path_1 = (price_list[c_1[1:]] - theta * price_list[c_1[:-1]])/(1-theta)
    
    id = i
    #apply the policy a few times
    for j in range(end_):
        c_2.append(int(mu_2[id]))
        id = int(mu_2[id])
    price_path_2 = (price_list[c_2[1:]] - theta * price_list[c_2[:-1]])/(1-theta)
    
    id = i
    #apply the policy a few times
    for j in range(end_):
        c_3.append(int(mu_3[id]))
        id = int(mu_3[id])
    price_path_3 = (price_list[c_3[1:]] - theta * price_list[c_3[:-1]])/(1-theta)
    
    id = i
    #apply the policy a few times
    for j in range(end_):
        c_4.append(int(mu_4[id]))
        id = int(mu_4[id])
    price_path_4 = (price_list[c_4[1:]] - theta * price_list[c_4[:-1]])/(1-theta)
    
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(start_+1,end_),price_path_1[start_:],\
            label = label_li[0], color = color_li[0], linestyle = linestyle_li[0],\
            marker = marker_li[0], mfc = 'none')
    ax.plot(np.arange(start_+1,end_),price_path_2[start_:],\
            label = label_li[1],color = color_li[1],linestyle = linestyle_li[1],\
            marker = marker_li[1],mfc = 'none')
    ax.plot(np.arange(start_+1,end_),price_path_3[start_:],\
            label = label_li[2],color = color_li[2],linestyle = linestyle_li[2],\
            marker = marker_li[2],mfc = 'none')
    ax.plot(np.arange(start_+1,end_),price_path_4[start_:],\
            label = label_li[3],color = color_li[3],linestyle = linestyle_li[3],\
            marker = marker_li[3],mfc = 'none')
    
    
    # plt.legend(bbox_to_anchor=(0., 1.1, 1., 0.1),ncol=2)
    ax.set_ylim(L,H+1)
    ax.set_xticks(np.arange(start_,end_+1,5))
    ax.set_xlabel('Time', size = 16)
    ax.set_ylabel('Price',size = 16)
    coeffname_r0 = coeffname + 'r0=' + str(round(price_list[i],2)).replace(".","dot")
    fig.savefig('./../pricing_output/%s_price_path'%coeffname_r0, dpi= 300, bbox_inches='tight')

#------------------------------------------------------------------------
#  plot cumulative revenue over time
#------------------------------------------------------------------------

#need at least 100 times to achieve convergence
end_ = 200
i = int((H/epsilon)/2)+1
c_1 = []
c_2 = []
c_3 = []
c_4 = []


id = i
#apply the policy a few times
for j in range(end_):
    c_1.append(int(mu_1[id]))
    id = int(mu_1[id])
price_path_1 = (price_list[c_1[1:]] - theta * price_list[c_1[:-1]])/(1-theta)

id = i
#apply the policy a few times
for j in range(end_):
    c_2.append(int(mu_2[id]))
    id = int(mu_2[id])
price_path_2 = (price_list[c_2[1:]] - theta * price_list[c_2[:-1]])/(1-theta)

id = i
#apply the policy a few times
for j in range(end_):
    c_3.append(int(mu_3[id]))
    id = int(mu_3[id])
price_path_3 = (price_list[c_3[1:]] - theta * price_list[c_3[:-1]])/(1-theta)

id = i
#apply the policy a few times
for j in range(end_):
    c_4.append(int(mu_4[id]))
    id = int(mu_4[id])
price_path_4 = (price_list[c_4[1:]] - theta * price_list[c_4[:-1]])/(1-theta)


revenue_1 = []
cum_revenue_1 = 0

revenue_2 = []
cum_revenue_2 = 0

revenue_3 = []
cum_revenue_3 = 0

revenue_4 = []
cum_revenue_4 = 0

for j in range(end_-1):
    temp_1 = R_uniform_fast(price_list[c_1[j]],price_path_1[j],((BH/2, BH/2, BH/2, BH/2), BH))
    # print(temp_1)
    
    cum_revenue_1 =cum_revenue_1 + gamma**j * temp_1
    revenue_1.append(cum_revenue_1)
    
    temp_2 = R_uniform_fast(price_list[c_2[j]],price_path_2[j],((BH/2, BH/2, BH/2, BH/2), BH))
    # print(temp_2)
    cum_revenue_2 =cum_revenue_2 + gamma**j * temp_2
    revenue_2.append(cum_revenue_2)
    
    temp_3 = R_uniform_fast(price_list[c_3[j]],price_path_3[j],((BH/2, BH/2, BH/2, BH/2), BH))
    # print(temp_2)
    cum_revenue_3 =cum_revenue_3 + gamma**j * temp_3
    revenue_3.append(cum_revenue_3)
    
    temp_4 = R_uniform_fast(price_list[c_4[j]],price_path_4[j],((BH/2, BH/2, BH/2, BH/2), BH))
    # print(temp_2)
    cum_revenue_4 =cum_revenue_4 + gamma**j * temp_4
    revenue_4.append(cum_revenue_4)
    



### plots
fig1, ax1 = plt.subplots()

ax1.plot(np.arange(start_+1,end_),revenue_1,label = label_li[0],color = color_li[0],\
         linestyle = linestyle_li[0],marker = marker_li[0],mfc = 'none',\
         markevery=10)
ax1.plot(np.arange(start_+1,end_),revenue_2,label = label_li[1],color = color_li[1],\
         linestyle = linestyle_li[1],marker = marker_li[1],mfc = 'none',\
         markevery=10)
ax1.plot(np.arange(start_+1,end_),revenue_3,label = label_li[2],color = color_li[2],\
         linestyle = linestyle_li[2],marker = marker_li[2],mfc = 'none',\
         markevery=10)
ax1.plot(np.arange(start_+1,end_),revenue_4,label = label_li[3],color = color_li[3],\
         linestyle = linestyle_li[3],marker = marker_li[3],mfc = 'none',\
         markevery=10)

plt.legend(bbox_to_anchor=(1.01, 0.6),loc = "upper left", ncol=1)
ax1.set_xticks(np.arange(start_,end_+1,20))
ax1.set_xlabel('Time', size = 16)
ax1.set_ylabel('Discounted Cumulative Revenue',size = 16)
coeffname_r0 = coeffname + 'r0=' + str(round(price_list[i],2)).replace(".","dot")
fig1.savefig('./../pricing_output/%s_cum_revenue'%coeffname_r0, dpi= 300, bbox_inches='tight')



