#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hanshengjiang
"""

from optimal_pricing_policy_exp_update import *
import random
import time

import sys

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
        
        
        

price_list = np.arange(L-epsilon,H+2*epsilon,epsilon)

if config == 'mmnl_1':
    coefficients = (1,1,1,1)
    weights = (1,0,0)
    demand_name = 'mmnl'
elif config == 'mmnl_2_1':
    coefficients = (-5,0,1,5)
    weights = (1,0,0)
    demand_name = 'mmnl'
elif config == 'mmnl_2_2':
    coefficients = (2,10,0.5,1)
    weights = (1,0,0)
    demand_name = 'mmnl'
elif config == 'mmnl_3':
    coefficients = (2,2,0.2,0.2,-1,0.2,0,0)
    weights = (0.5,0.5,0)
    demand_name = 'mmnl'
elif config == 'mmnl_3_1':
    coefficients = (2,2,0.2,0.2)
    weights = (1,0,0)
    demand_name = 'mmnl'
elif config == 'mmnl_3_2':
    coefficients = (-1,0.2,0,0)
    weights = (1,0,0)
    demand_name = 'mmnl'
elif config == 'mmnl_myopic':
    coefficients = (2,2,0.2,0.2)
    weights = (1,0,0)
    demand_name = 'mmnl'

# name prefix for saved plots
coeffname = 'inf_hor_fixed_price_range'+str(L).replace(".","dot")+'_'+str(H).replace(".","dot")+'_'+str(T)+'_'\
+str(coefficients).replace(" ", "_").replace(".","dot")\
+str(weights).replace(" ","_").replace(".","dot")+'_'+str(theta).replace(".", "dot")+'_'+str(gamma).replace(".", "dot")+demand_name




#---------------------------------#
# optimal pricing policy
start_time = time.time()
#optimal pricing policy

V,mu \
= inf_hor_pricing_pricerange(L,H,theta,epsilon,np.inf,gamma,coefficients,weights,demand_name)

print("running time {}".format(time.time()- start_time))
#---------------------------------#



 


#---------------------------------#
# plot results
#---------------------------------#


#---------------reference price transition function-------------------------#
fig, ax = plt.subplots()
plt.tight_layout()
#ax.plot(price_list[np.arange(len(mu)).astype(int)], price_list[mu.astype(int)],\
#       color = 'black');
ax.scatter(price_list[np.arange(len(mu)).astype(int)], price_list[mu.astype(int)],\
        color = 'black',marker = 'o', facecolors = 'none', s = 5);  # s controls markersize          
ax.set_xlabel(r"$r_t$",size = 16);
ax.set_ylabel(r"$r_{t+1}$",size = 16);
ax.set_ylim(L-1,H+1)
fig.savefig('./../pricing_output/%s_referenceprice'%coeffname, dpi= 300)

#---------------value function------------------------#
fig1, ax1 = plt.subplots()
plt.tight_layout()
ax1.plot(price_list[np.arange(len(mu)).astype(int)], \
         V, color = 'black');
ax1.set_xlabel(r"$r$",size = 16);
ax1.set_ylabel(r"$V^*(r)$",size = 16);
#ax2.set_ylim(,)
fig1.savefig('./../pricing_output/%s_value_function'%coeffname, dpi= 300)

#---------------pricing policy function-------------------------#
fig2, ax2 = plt.subplots()
plt.tight_layout()
#ax2.plot(price_list[np.arange(len(mu)).astype(int)], 
 #        (price_list[mu.astype(int)] - theta * price_list[np.arange(len(mu)).astype(int)])/(1-theta),\
  #       color = 'black');
ax2.scatter(price_list[np.arange(len(mu)).astype(int)], 
         (price_list[mu.astype(int)] - theta * price_list[np.arange(len(mu)).astype(int)])/(1-theta),\
         color = 'black', marker = 'o', s= 5);
ax2.set_xlabel(r"$r$",size = 16);
ax2.set_ylabel(r"$p^*(r)$",size = 16);
ax2.set_ylim(L-1,H+1)
fig2.savefig('./../pricing_output/%s_price'%coeffname, dpi= 300)


#---------------price path-------------------------#
# plot price path
start_ = 0 # staring time for plots
end_ = 20
i_li = random.sample(range(len(V)),10)
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
    
    fig3, ax3 = plt.subplots()
    plt.tight_layout()
    price_path = (price_list[c[1:]] - theta * price_list[c[:-1]])/(1-theta)
    ax3.plot(np.arange(start_,end_),price_path[start_:],color = 'black',linestyle = '--')
    ax3.scatter(np.arange(start_,end_),price_path[start_:],marker = 'o',facecolors = 'none',edgecolors = 'black')
    ax3.set_ylim(L,H+1)
    ax3.set_xticks(np.arange(start_,end_+1,5))
    ax3.set_xlabel('Time', size = 16)
    #ax3.set_ylabel('optimal price path',size = 16)
    ax3.set_ylabel('Price',size = 16)
    coeffname_r0 = coeffname + 'r0=' + str(round(price_list[i],2)).replace(".","dot")
    fig3.savefig('./../pricing_output/%s_price_path'%coeffname_r0, dpi= 300)




#---------------long term price path plot-------------------------#

id = i
c = []
start_ = 80
end_ = 100

# need to add the first reference price
c.append(id)

for j in range(end_):
    c.append(int(mu[id]))
    id = int(mu[id])

fig4, ax4 = plt.subplots()
plt.tight_layout()
price_path = (price_list[c[1:]] - theta * price_list[c[:-1]])/(1-theta)
ax4.plot(np.arange(start_,end_),price_path[start_:],color = 'black',linestyle = '--')
ax4.scatter(np.arange(start_,end_),price_path[start_:],marker = 'o',facecolors = 'none',edgecolors = 'black')
ax4.set_ylim(L,H+1)
ax4.set_xticks(np.arange(start_,end_+1,5))
ax4.set_xlabel('Time', size = 16)
#ax4.set_ylabel('optimal price path',size = 16)
ax4.set_ylabel('Price',size = 16)
coeffname_r0 = coeffname + 'r0=' + str(round(price_list[i],2)).replace(".","dot")+'100time'
fig4.savefig('./../pricing_output/%s_longterm_price_path'%coeffname_r0, dpi= 300)


#--------------revenue plot for heterogeneous market --------------------#
c = []
id = int((H/epsilon)/2)+1

# need to add the first reference price
c.append(id)
#### plot revenues
for j in range(20):
    c.append(int(mu[id]))
    id = int(mu[id])
fig3, ax3 = plt.subplots(figsize = (10,5))
price_path = (price_list[c[1:]] - theta * price_list[c[:-1]])/(1-theta)
r1 = []
r2 = []
for j in range(20):
    r = price_list[c[j]]
    p = price_path[j]
    print("================{}================".format(j))
    print(r,p,coefficients,weights[0])
    temp1 = R(r,p,coefficients[:4],(weights[0],0,0))
    print("temp1", temp1)
    r1.append(temp1)
    temp2 = R(r,p,coefficients[4:],(weights[1],0,0))
    print("temp2", temp2)
    r2.append(temp2)
r = np.array(r1) + np.array(r2)
ax3.plot(np.arange(20),r,label = r'Total',marker = 'o',mfc = 'none',linestyle = '-',color = 'tab:gray')
ax3.plot(np.arange(20),r1,label = r'Consumer $A$',marker = 'd',mfc = 'none',linestyle = '--',color= 'tab:blue')
ax3.plot(np.arange(20),r2,label = r'Consumer $B$', marker = 's',mfc = 'none', linestyle = ':',color = 'tab:red')
# ax3.scatter(np.arange(20),price_path[start_:],marker = 'o',facecolors = 'none',edgecolors = 'black')
#ax3.set_ylim(L,H+1)
ax3.set_xticks(np.arange(0,21,5))
ax3.set_xlabel('Time', size = 16)
ax3.set_ylabel('Expected Revenue',size = 16)
plt.legend(bbox_to_anchor=(1.02, 1))
plt.tight_layout()
coeffname_r0 = coeffname + 'r0=' + str(round(price_list[i],2)).replace(".","dot")
fig3.savefig('./../pricing_output/%s_revenue'%coeffname_r0, dpi= 300)
