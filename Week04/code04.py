import scipy as sp
import statistics
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os
#scipy.optimize()
path = os.getcwd()



#Problem1
def simulation(n, method):
    np.random.seed(234)
    r_t = np.random.normal(0, 1, n)

    p_t_1 = 100
    #p_t = r_t + p_t_1
    p_t = [0]*len(r_t)
    if method == 'CBM':
        p_t = p_t_1 + r_t
    elif method == 'ARS':
        p_t = p_t_1 *(r_t + 1) 
    elif method == 'LR':
        p_t = p_t_1 * np.exp(r_t)       

    mean = sum(p_t) / len(p_t)  #82.11492745544753
    std = math.sqrt(sum((x-mean)**2 for x in p_t) / (len(p_t)-1)) 
    return [mean, std]

simulation(10000, 'CBM') #[99.99807992497432, 0.9983846342220718]
simulation(10000, 'ARS') #[98.99937707258351, 101.87680798705381]
simulation(10000, 'LR')  #[166.83023248105218, 220.90530738215818]



#Problem2
df0 = pd.read_csv(path+'/Week04/DailyPrices.csv')
#a. Use DailyPrices.csv. Calculate the arithmetic returns for all prices.
def ret_discrete(df,column):
    for i in range(len(df)-1):
        df[column][i] = df[column][i+1]/df[column][i]-1
    df[column] = df[column].shift(periods=1, fill_value=0)

def ret_log(df,column):
    for i in range(len(df)-1):
        df[column][i] = math.log(df[column][i+1]/df[column][i])
    df[column] = df[column].shift(periods=1, fill_value=0)

def return_calculate(df0, method="DISCRETE"):
    df3 = df0.copy()
    df3.set_index('Date', inplace = True)
    col = df3.columns

    if method == "DISCRETE":
        for i in col:
            ret_discrete(df3,i)       
    elif method == "LOG":
        for i in col:
            ret_log(df3,i)
    
    return df3
        
df_return_dis = return_calculate(df0,'DISCRETE')

#b. Remove the mean from the series so that the mean(META)=0
meta = df_return_dis['META']
mean = np.average(meta)
meta_removed = meta - mean
mean_removed = np.average(meta_removed) #-5.217213461584382e-19

#c. Calculate VaR
# Normal Distribution
std = np.std(meta_removed)
z_score = 1.65
VaR = std * z_score * df0['META'][len(df0)-1]#16.225807942951008

# Normal Distribution with an Exponentially Weighted variance(0.94)
def populateWeights(λ,data):
    n = len(data)
    total_weights = 0.0
    w =[]
    for i in range(n):
        w_i = (1-λ) * (λ**(n-1-i))
        total_weights += w_i
        w.append(w_i)
    #print(len(w))
    #normalize weights to 1
    w = [i/total_weights for i in w]
    c_w = [sum(w[0:i]) for i in range(len(w))]
    weights = pd.DataFrame(w,columns = ['weights'])
    cum_weights = pd.DataFrame(c_w,columns = ["λ="+str(λ)])
    return w
weights_ = populateWeights(0.94, meta_removed)
#cumu_weights = [sum(weights_[0:i]) for i in range(1, len(weights_)+1)]
weighted_return = weights_ * meta_removed
std_weighted = np.std(weighted_return)
z_score = 1.65
VaR_weighted = std_weighted * z_score * df0['META'][len(df0)-1] #1.5564

############################################################################################################
λ = 0.94
weighted_var = meta_removed.ewm(alpha = 1 - λ).var().iloc[-1]
#print("weighted_var: ", weighted_var)

VaR_ew = - norm.ppf(1 - λ, np.mean(meta_removed), np.sqrt(weighted_var)) * df0['META'][len(df0)-1]
print("VaR_ew: ", VaR_ew)

############################################################################################################

# MLE fitted T distribution
from scipy.optimize import minimize
from scipy.stats import t
'''
v = len(meta_removed)-1 
data = meta_removed
def neg_log_likelihood(params):
    nu, sigma = params
    log_likelihood = -np.sum(t.logpdf(data, df=nu, loc=0, scale=sigma))
    return log_likelihood
ll_model_t = sp.optimize.minimize(neg_log_likelihood, np.array([1,1]))

nu_mle, sigma_mle = ll_model_t.x
print("MLE Estimates:")
print("Standard Deviation (sigma):", sigma_mle)
print("Degrees of Freedom (nu):", nu_mle)

alpha = 0.05
t_score = t.ppf(1 - alpha/2, round(nu_mle))
VaR_t = (mean+t_score * std) * df0['META'][len(df0)-1]
'''

def calculate_var(prices, confidence_level=0.05):
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Fit the T-distribution
    params = t.fit(returns)
    
    # Calculate VaR
    VaR = t.ppf(confidence_level, *params)
    
    return VaR

VaR_95 = calculate_var(meta_removed, 0.05)
print(f"95% VaR: {VaR_95}")
# 95% VaR: -7.531381097807715

# Fitted AR(1) model
import numpy as np
import statsmodels.api as sm

def calculate_var_ar1(returns, confidence_level, num_simulations):
    # Fit AR(1) model to historical returns
    ar1_model = sm.tsa.ARIMA(returns, order=(1, 0, 0)).fit()

    # Extract AR(1) parameters
    mu = ar1_model.params[0]
    phi = ar1_model.arparams[0]
    sigma = ar1_model.params[2]

    # Simulate future returns
    simulated_returns = [mu]
    for _ in range(num_simulations):
        simulated_return = mu + phi * (simulated_returns[-1] - mu) + sigma * np.random.randn()
        simulated_returns.append(simulated_return)

    # Calculate VaR from simulated returns
    simulated_returns = np.array(simulated_returns)
    var = np.percentile(simulated_returns, 100 - confidence_level * 100)

    return var * df0['META'][len(df0)-1]

confidence_level = 0.95
num_simulations = 10000
var = calculate_var_ar1(meta_removed, confidence_level, num_simulations)
print(f"Value at Risk (VaR) at {confidence_level*100:.2f}% confidence level: {var:.4f}")
#Value at Risk (VaR) at 95.00% confidence level: -0.5362

# Historic Simulation
import numpy as np

def calculate_var_historical(returns, confidence_level):
    # Sort returns in ascending order
    returns_sorted = np.sort(returns)

    # Determine VaR observation index
    index = int(len(returns_sorted) * (1 - confidence_level))

    # Calculate VaR
    var = -returns_sorted[index] *df0['META'][len(df0)-1]

    return var

confidence_level = 0.95
# Calculate VaR using historical simulation
var = calculate_var_historical(meta_removed, confidence_level)
print(f"Value at Risk (VaR) at {confidence_level*100:.2f}% confidence level: {var:.4f}")
#Value at Risk (VaR) at 95.00% confidence level: 11.8128

# Comparison of 5 methods






#Problem3
import pandas as pd
import numpy as np
from numpy.linalg import eigvals
from scipy.stats import norm

df1 = pd.read_csv(path+'/portfolio.csv')

def filter_data(portfolio):
    holdings_1 = df1[df1['Portfolio']== portfolio][['Stock', 'Holding']].reset_index()
    holdings_1.drop('index',axis = 1,inplace = True)
    hold1_stock = list(holdings_1['Stock'])
    holdings1 = {}
    for i in range(len(holdings_1)):
        holdings1[holdings_1['Stock'][i]] = holdings_1['Holding'][i]
    hold1_price = df0[hold1_stock]
    return [hold1_price, hold1_stock, holdings1]
para1 = filter_data('A')
para2 = filter_data('B')
para3 = filter_data('C')


# 1. EWM
def cal_MC_VaR(prices, common_stocks, holdings, df_return_dis0):
    LAMBDA = 0.94
    num = len(common_stocks)
    ewm_cov = df_return_dis[common_stocks].ewm(span=(2/(1-LAMBDA))-1).cov()[-num:]
    # Filter prices and returns
    returns = df_return_dis0[common_stocks]
    current_prices = prices.iloc[-1]
    # Calculate portfolio value
    PV = 0.0
    delta = np.zeros(len(common_stocks))

    for i, s in enumerate(common_stocks):
        value = holdings[s] * current_prices[s]
        PV += value
        delta[i] = value

    delta /= PV

    Sigma = np.cov(returns, rowvar=False)
    p_sig = np.sqrt(delta @ ewm_cov @ delta.T)
    VaR = -PV * norm.ppf(0.05) * p_sig
    print(delta)
    print("Delta Normal")
    print("Current Portfolio Value:", PV)
    print("Current Portfolio VaR:", VaR)
    print()

''' 
    # MC VaR - Same Portfolio
    n = 10000
    np.random.seed(234)
    sim_returns = np.random.multivariate_normal(np.zeros(len(common_stocks)), Sigma, n)

    sim_prices = (1 + sim_returns) * np.array(current_prices)
    vHoldings = np.array([holdings[s] for s in common_stocks])

    pVals = sim_prices @ vHoldings
    pVals.sort()

    VaR = PV - pVals[int(0.05 * n)]
    print("MC Normal")
    print("Current Portfolio Value:", PV) #1089316.15994
    print("Current Portfolio VaR:", VaR) #21135.25899507571
    print()

    # Historical VaR
    sim_prices = (1 + returns.values) * np.array(current_prices[common_stocks])
    pVals = sim_prices @ np.array([holdings[s] for s in common_stocks])
    pVals.sort()

    n = len(returns)
    VaR = PV - pVals[int(0.05 * n)]
    print("Historical VaR ")
    print(f"Current Portfolio Value: {PV}")
    print(f"Current Portfolio VaR: {VaR}")
'''
# Weighted
df_return_dis0 = df_return_dis.copy()
for i in df_return_dis0.columns:
    df_return_dis0[i] = weights_ * df_return_dis0[i]
cal_MC_VaR(para1[0],para1[1],para1[2],df_return_dis0)
cal_MC_VaR(para2[0],para2[1],para2[2],df_return_dis0)
cal_MC_VaR(para3[0],para3[1],para3[2],df_return_dis0)

# Normal
def cal_MC_VaR(prices, common_stocks, holdings, df_return_dis0):
    # Filter prices and returns
    returns = df_return_dis[common_stocks]
    current_prices = prices.iloc[-1]
    # Calculate portfolio value
    PV = 0.0
    delta = np.zeros(len(common_stocks))

    for i, s in enumerate(common_stocks):
        value = holdings[s] * current_prices[s]
        PV += value
        delta[i] = value

    delta /= PV

    Sigma = np.cov(returns, rowvar=False)
    p_sig = np.sqrt(delta @ Sigma @ delta.T)
    VaR = -PV * norm.ppf(0.05) * p_sig

    print("Delta Normal")
    print("Current Portfolio Value:", PV)
    print("Current Portfolio VaR:", VaR)
    print()

    # MC VaR - Same Portfolio
    n = 10000
    np.random.seed(234)
    sim_returns = np.random.multivariate_normal(np.zeros(len(common_stocks)), Sigma, n)

    sim_prices = (1 + sim_returns) * np.array(current_prices)
    vHoldings = np.array([holdings[s] for s in common_stocks])

    pVals = sim_prices @ vHoldings
    pVals.sort()

    VaR = PV - pVals[int(0.05 * n)]
    print("MC Normal")
    print("Current Portfolio Value:", PV) #1089316.15994
    print("Current Portfolio VaR:", VaR) #21135.25899507571
    print()

    # Historical VaR
    sim_prices = (1 + returns.values) * np.array(current_prices[common_stocks])
    pVals = sim_prices @ np.array([holdings[s] for s in common_stocks])
    pVals.sort()

    n = len(returns)
    VaR = PV - pVals[int(0.05 * n)]
    print("Historical VaR ")
    print(f"Current Portfolio Value: {PV}")
    print(f"Current Portfolio VaR: {VaR}")

# Weighted
df_return_dis0 = df_return_dis.copy()
for i in df_return_dis0.columns:
    df_return_dis0[i] = weights_ * df_return_dis0[i]
cal_MC_VaR(para1[0],para1[1],para1[2],df_return_dis0)
cal_MC_VaR(para2[0],para2[1],para2[2],df_return_dis0)
cal_MC_VaR(para3[0],para3[1],para3[2],df_return_dis0)







