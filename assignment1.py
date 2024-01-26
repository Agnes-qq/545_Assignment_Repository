import scipy as sp
import statistics
import pandas as pd
import numpy as np
import math
#scipy.optimize()

##########################################################################################
# Problem 1
# mean, variance, kurtosis, skew
df = pd.read_csv("/Users/ruoqiyan/Desktop/545/Week02/problem1.csv")
# a: using formula
n = len(df['x'])
mean = sum(df['x']) / n #1.0489703904839582
biased_variance = sum((x-mean)**2 for x in df['x']) / n #5.421793461199844
unbiased_variance = sum((x-mean)**2 for x in df['x']) / (n-1) #5.4272206818817255
unbiased_skewness_normalized = (sum((x-mean)**3 for x in df['x']) * (n/((n-1)*(n-2)))) / unbiased_variance**(3/2) #0.8819320922598407
biased_kurtosis_normalized = (sum((x-mean)**4 for x in df['x']) / n) / (unbiased_variance**2) #26.06998251061053
biased_kurtosis = ( sum( (x-mean)**4 for x in df['x'] ) / n ) 
unbiased_kurtosis_normalized = ( (n**2) / ( ((n-1)**3) * ((n-2)**2-3*n+3) ) ) *                                                \
                                ( ( n * ( (n-1)**2 ) + (6*n-9) ) * biased_kurtosis - n*(6*n-9)*(biased_variance**2) )                 \
                                / (unbiased_variance**2)
# b: using package
print("mean: ", sp.stats.describe(df['x'])[2]) #1.0489703904839585
print("variance: ", sp.stats.describe(df['x'])[3]) #5.427220681881728
print("skewness: ", sp.stats.skew(df['x'])) #0.8806086425277364
print("kurtosis: ", sp.stats.kurtosis(df['x'])) #23.122200789989723


# c: unbiased


'''
So we can see that the variance, skewness and kurtosis produced by the scipy package is unbiased
'''
##########################################################################################

##########################################################################################
# Problem 2
df2 = pd.read_csv("/Users/ruoqiyan/Desktop/545/Week02/problem2.csv")
y = df2['y']
x = df2["x"]
# a 
# ols
import statsmodels.api as sm
x = sm.add_constant(x)
model = sm.OLS(y, x)
results = model.fit()
res_params = results.params
print("function: y = " + str(res_params[1]) + " * x + (" + str(res_params[0]) + ")") # function: y = 0.7752740987226112 * x + (-0.08738446427005077)
y_pre = res_params[1]*x + res_params[0]
err = y_pre - y
mean_err = sum(err)/len(err)
std_err = sum( (e - mean_err)** 2 for e in err) / (len(x)-1) # 1.0125896972573178
print("ols error std: ", std_err) 

# mle
def Likelihood_func(parameters): # function for minimization on condition of normal distribution
    m = parameters[0]
    b = parameters[1]
    sigma = parameters[2]
    for i in np.arange(0, len(x)):
        y_exp = m * x + b
    L = (len(x)/2 * np.log(2 * np.pi) + len(x)/2 * np.log(sigma ** 2) + 1 /
         (2 * sigma ** 2) * sum((y - y_exp) ** 2))
    return L
ll_model_n = sp.optimize.minimize(Likelihood_func, np.array([1,1,1]), method = 'L-BFGS-B')
print("function: y = " + str(ll_model_n.x[0]) + " * x + (" + str(ll_model_n.x[1]) + ")") # function: y = 0.7752749227004981 * x + (-0.087380649693968)
print('sigma: ', ll_model_n.x[2]) # sigma:  1.0037545081200103
# Result 
'''
These two functions produced same beta, but the sigma for MLE method is smaller than the std_err of ols, 
which resulted from the fact that the estimator of MLE not unbiased so it is divided by a larger denominator
'''

# b
v = len(x)-1
def Likelihood_t(parameters): # function for minimization on condition of t-distribution
    m = parameters[0]
    b = parameters[1]
    nu = parameters[2] # degree of freedom
    sigma = parameters[3]
    for i in np.arange(0, len(x)):
        y_exp = m * x + b
    L = -np.sum(sp.stats.t.logpdf(y-y_exp, df=nu, loc=0, scale=sigma))
    return L
ll_model_t = sp.optimize.minimize(Likelihood_t, np.array([1,1,1,1]), method = 'L-BFGS-B')
print("function: y = " + str(ll_model_t.x[0]) + " * x + (" + str(ll_model_t.x[1]) + ")") # function: y = 0.6750098844824576 * x + (-0.09726916094951703)
print('sigma: ', ll_model_t.x[3]) # sigma:  0.8551059789405173
# compare
y_pre_n = ll_model_n.x[0]*x + ll_model_n.x[1]
err_n = y_pre_n - y
sst_n = np.sum( (y-np.mean(y))**2 )
ssr_n = np.sum(err_n**2)
r_square_n = 1- ssr_n/sst_n  #0.34560688355471014

y_pre_t = ll_model_t.x[0]*x + ll_model_t.x[1]
err_t = y_pre_t - y
sst_t = np.sum( (y-np.mean(y))**2 )
ssr_t = np.sum(err_t**2)
r_square_t = 1- ssr_t/sst_t  #0.3396547950063461
'''
The t-squared under mle for normal distribution assumption is higher than the t-distribution, so the formal one would be the better fit. 
'''

# c


##########################################################################################

##########################################################################################
# Problem 3
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_process import ArmaProcess
df3 = pd.read_csv("/Users/ruoqiyan/Desktop/545/Week02/problem3.csv")
#y_t =  + .05*e_t-1 + e, e ~ N(0,.01)