import scipy as sp
import statistics
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
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
biased_skewness_normalized = (sum((x-mean)**3 for x in df['x']) / (n)) / unbiased_variance**(3/2)
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

sp.stats.var(df['x'])

# c: unbiased
sample_size = 100
samples = 1000
rng = np.random.default_rng()

kurts= []
for i in range(samples):
    kurt = sp.stats.kurtosis(np.random.normal(0,1,size=sample_size))
    kurts.append(kurt)


mean_kurts = sum(kurts) / len(kurts)
var = sum((i-mean_kurts)**2 for i in kurts) / (len(kurts)-1)
t = mean_kurts/math.sqrt(var/samples) 
p_value = 2 * (1 - sp.stats.t.cdf(abs(t), df=samples - 1))
print(f"Mean of Kurtosis: {mean_kurts}")
print(f"Variance of Kurtosis: {var}")
print(f"p-value: {p_value}")


ttest_result = sp.stats.ttest_1samp(kurts, 0)  # Testing against kurtosis of 0 (normal distribution)
p_value_2 = ttest_result.pvalue
print(f"Built-in t-test p-value: {p_value_2}")
print(f"Match the stats package test?: {np.isclose(p_value, p_value_2)}")
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
df2_x = pd.read_csv("/Users/ruoqiyan/Desktop/545/Week02/problem2_x.csv")
x1 = df2_x['x1']
x2 = df2_x['x2']
x = np.column_stack((x1, x2))
# mle estimators
mean_vector = np.mean(x, axis = 0)
covariance_matrix = np.cov(x, rowvar=False)
multi_normal = sp.stats.multivariate_normal(mean_vector, covariance_matrix) # X distribution

#conditional distributions
df2_x1 = pd.read_csv("/Users/ruoqiyan/Desktop/545/Week02/problem2_x1.csv")

def conditional_distribution(x1, mean_vector, covariance_matrix):
    mean_x1, mean_x2 = mean_vector
    cov_11, cov_12, cov_21, cov_22 = covariance_matrix[0, 0], covariance_matrix[0, 1],  \
                                     covariance_matrix[1, 0], covariance_matrix[1, 1]

    mean_cond = mean_x2 + cov_21 / cov_11 * (x1 - mean_x1)
    var_cond = cov_22 - cov_21 * cov_12 / cov_11

    return mean_cond, var_cond

x1_values = np.array(x1)
means = []
conf_intervals = []

for x_1 in x1_values:
    mean_cond, var_cond = conditional_distribution(x_1, mean_vector, covariance_matrix)
    std_cond = np.sqrt(var_cond)

    # 95% confidence interval
    conf_int = sp.stats.norm.interval(0.95, loc=mean_cond, scale=std_cond)

    means.append(mean_cond)
    conf_intervals.append(conf_int)

# Plot
plt.fill_between(x1_values, [ci[0] for ci in conf_intervals], [ci[1] for ci in conf_intervals], color='gray', alpha=0.5)
plt.plot(x1_values, means, label='Expected_x2')
plt.xlabel('x1')
plt.ylabel('Expected_x2')
plt.title('Expected_x2')
plt.legend()
plt.show()

##########################################################################################

##########################################################################################
# Problem 3
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
df3 = pd.read_csv("/Users/ruoqiyan/Desktop/545/Week02/problem3.csv")
x = df3['x']

acf_values = acf(x, nlags = 10)
plot_acf(x, lags=10)
plt.tight_layout()
plt.show()

pacf_values = pacf(x, nlags = 10)
plot_pacf(x, lags=10)
plt.tight_layout()
plt.show()
'''
acf_values([ 1.        ,  0.20169955, -0.2953451 ,  0.27087207,  0.36085191,
            -0.161369  , -0.13072282,  0.21078742,  0.09021041, -0.19982537,
            -0.05712463])
'''
# AR(1): x_t = b + a * x_t-1 + w_t
model_ar1 = sm.tsa.ARIMA(x, order=(1, 0, 0))
results_ar1 = model_ar1.fit()
res_ar1 = results_ar1.params
print("function: x_t = " + str(res_ar1[1]) + " * x_t-1 + (" + str(res_ar1[0]) + ")") 
# function: x_t = 0.2018990252391578 * x_t-1 + (2.125792817985664)
aic_ar1 = results_ar1.aic #1644.6555047688475

# AR(2): x_t = b + a1 * x_t-1 + a2 * x_t-2 + w_t
model_ar2 = sm.tsa.ARIMA(x, order=(2, 0, 0))
results_ar2 = model_ar2.fit()
res_ar2 = results_ar2.params
print("function: x_t = " + str(res_ar2[2]) + " * x_t-2 + " + str(res_ar2[1]) + " * x_t-1 + (" + str(res_ar2[0]) + ")") 
# function: x_t = -0.3505147997938495 * x_t-2 + 0.273164912108471 * x_t-1 + (2.1270087995492126)
aic_ar2 = results_ar2.aic #1581.079265904978

# AR(3): x_t = b + a1 * x_t-1 + a2 * x_t-2 + w_t
model_ar3 = sm.tsa.ARIMA(x, order=(3, 0, 0))
results_ar3 = model_ar3.fit()
res_ar3 = results_ar3.params
print("function: x_t = " + str(res_ar3[3]) + " * x_t-3 + " + str(res_ar3[2]) + " * x_t-2 + " + str(res_ar3[1]) + " * x_t-1 + (" + str(res_ar3[0]) + ")") 
# function: x_t = 0.5047427118503807 * x_t-3 + -0.48870825282956476 * x_t-2 + 0.4515020775436772 * x_t-1 + (2.1209226708550624)
aic_ar3 = results_ar3.aic #1436.6598066945867

# MA(1): x_t = m + e_t + a * e_t-1 , e_t ~ N(0,.01)
model_ma1 = sm.tsa.ARIMA(x, order=(0, 0, 1))
results_ma1 = model_ma1.fit()
res_ma1 = results_ma1.params
print("function: x_t = " + str(res_ma1[1]) + " * e_t-1 + (" + str(res_ma1[0]) + ")") 
# function: x_t = 0.643445172084989 * e_t-1 + (2.123607598768106)
aic_ma1 = results_ma1.aic #1567.4036263707874

# MA(2): x_t = m + e_t + a1 * e_t-1 + a2 * e_t-2 , e_t ~ N(0,.01)
model_ma2 = sm.tsa.ARIMA(x, order=(0, 0, 2))
results_ma2 = model_ma2.fit()
res_ma2 = results_ma2.params
print("function: x_t = " + str(res_ma2[2]) + " * e_t-2 + "+ str(res_ma2[1]) + " * e_t-1 + (" + str(res_ma2[0]) + ")") 
# function: x_t = -0.23063118323240853 * e_t-2 + 0.43444893992255124 * e_t-1 + (2.1254978458712155)
aic_ma2 = results_ma2.aic #1537.9412063807388

# MA(3): x_t = m + e_t + a1 * e_t-1 + a2 * e_t-2 + a3 * e_t-3 , e_t ~ N(0,.01)
model_ma3 = sm.tsa.ARIMA(x, order=(0, 0, 3))
results_ma3 = model_ma3.fit()
res_ma3 = results_ma3.params
print("function: x_t = " + str(res_ma3[3]) + " * e_t-3 + " + str(res_ma3[2]) + " * e_t-2 + " + str(res_ma3[1]) + " * e_t-1 + (" + str(res_ma3[0]) + ")") 
# function: x_t = -0.15312039585565979 * e_t-3 + -0.2286227115905812 * e_t-2 + 0.5581589507970652 * e_t-1 + (2.12586661823169)
aic_ma3 = results_ma3.aic #1536.8677087350309

'''
AR(3) is the best among these models with the lowest AIC.
'''

from pandas import read_csv
from matplotlib import pyplot

x.plot()
pyplot.show()

