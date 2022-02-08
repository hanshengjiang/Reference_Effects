import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd

import sys


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please provide experiments configuration!")
    
    #sys_argv[0] is the name of the .py file
    num_component = int(sys.argv[1])
    seed_id = sys.argv[2]
    folder_path = sys.argv[3]
        
def EM_alg(x, y, k, pi=None, beta=None, r_seed=626, max_iters=100, beta_tol=1e-2):
    '''
    Input
    -----
    x (n,p) float
    y (n,) float
    k scalar integer
    '''
    
    DEBUG = 0
    
    if DEBUG==0:
        np.seterr(over='ignore') # Suppress overflow warnings. Overflows are taken care of in the code
    
    np.random.seed(r_seed)
    
    n = np.shape(x)[0]
    p = np.shape(x)[1]
    
    # Initialize parameters
    if pi==None:
        pi = np.random.random(k)
        pi = pi / np.sum(pi) # sum of probabilities should be 1
    if beta==None:
        beta = np.reshape(np.random.randn(p*k), [k, p])
    w = np.zeros([k,n])
    
    for step in range(max_iters):
        if np.mod(step,5)==0:
            print('iter: ' + str(step))
            
        # E step
        for i in range(n):
            for j in range(k):
                denom = 0
                if y[i]==1:
                    for jj in range(k):
                        temp = np.exp(np.sum(beta[jj]*x[i]))
                        if temp==np.inf: # check if overflow
                            denom = denom + pi[jj] # if overflow, then temp/(1+temp) ~= 1
                        else:
                            denom = denom + pi[jj] * temp / (1 + temp)
                    temp = np.exp(np.sum(beta[j]*x[i]))
                    if temp==np.inf:
                        w[j,i] = pi[j] / denom
                    else:
                        w[j,i] = (pi[j] * temp / (1 + temp)) / denom
                elif y[i]==0:
                    for jj in range(k):
                        temp = np.exp(np.sum(beta[jj]*x[i]))
                        if temp==np.inf: # check if overflow
                            denom = denom
                        else:
                            denom = denom + pi[jj] / (1 + temp)
                    temp = np.exp(np.sum(beta[j]*x[i]))
                    if temp==np.inf:
                        w[j,i] = 0
                    else:
                        w[j,i] = (pi[j] / (1 + temp)) / denom
                else:
                    print("error: y is not binary")

        # M step

        # Update pi
        pi_temp = np.copy(pi)
        for j in range(k):
            pi[j] = np.sum(w[j]) / n
        if DEBUG:
            print('pi difference is ' + str(np.sum(np.power((pi_temp - pi), 2))))
        
        # Update beta
        beta_temp = np.copy(beta)
        for j in range(k):
            clf = LogisticRegression(random_state = r_seed,
                                     penalty = 'none',     # There is regularization by default, but removing it didn't seem to do much
                                     multi_class = 'auto', # Changing this to auto from multinomial helped
                                     solver = 'lbfgs',
                                     fit_intercept = False).fit(x,y,sample_weight=w[j])
            beta[j,:] = clf.coef_
        
        beta_diff = np.sum(np.power((beta_temp - beta), 2))
        if DEBUG:
            print('beta difference is ' + str(beta_diff))
        if beta_diff < beta_tol:
            print('convergence criteria reached')
            break
        
    # sort results
    idx = np.argsort(-pi)
    return pi[idx], beta[idx]
    
#---------------sanity check with simulated data-----#       
#r_seed = 626
#np.random.seed(r_seed)
#k=2
#p=2
#n=5000
#x = np.reshape(np.random.randn(n*p), [n,p])
##beta = np.reshape(np.random.randn(k*p), [k,p])
#beta = -5*np.ones([k,p])
#pi = np.random.random(k)
#pi = pi / np.sum(pi) # sum of probabilities should be 1
#
#y = np.zeros(n)
#for i in range(n):
#    rdn = np.random.uniform(0,1,1)
#    for j in range(k):
#        if rdn >= pi[:j].sum() and rdn < pi[:j+1].sum():
#            beta_s = beta[j]
#    p = np.exp(np.sum(beta_s*x[i])) / (1 + np.exp(np.sum(beta_s*x[i])))
#    rdn_y = np.random.uniform(0,1,1)
#    if rdn_y <= p:
#        y[i] = 1
#    else:
#        y[i] = 0
#        
#for i in range(5):
#    print(EM_alg(x, y, k, r_seed=r_seed*i))
#----------------------------------------# 

X = pd.read_csv(folder_path + '/X.csv', header = None).values
y = pd.read_csv(folder_path + '/y.csv', header = None).values.ravel()
  
pi, beta = EM_alg(X, y, num_component)
       
pd.DataFrame(pi).to_csv(folder_path + '/alpha_EM.csv'\
                    , index= False, header = False)
pd.DataFrame(beta.T).to_csv(folder_path +'/B_EM.csv'\
                    , index = False, header = False)












