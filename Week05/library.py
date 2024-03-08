import scipy as sp
import statistics
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import eig
import os
from scipy.stats import norm
from scipy.linalg import cholesky, LinAlgError
import numpy.random as npr
#scipy.optimize()
path = '/Users/ruoqiyan/Documents/FinTech545_Spring2024/testfiles/data/'



def test(res, res_out, tolerance):
    res = pd.DataFrame(res)
    for i in range()
    if abs(res - res_out) < tolerance:
        return True
    else:
        return False

#1. Covariance estimation techniques.
#Test 1 - Pearson
x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/test1.csv")
def missing_cov(x, skipmiss, func = 'cov'):
    if skipmiss == True: #skip na
        x_ = x[pd.isna(x).any(axis = 1) == False]
        if func == 'cov':
            res = np.cov(x_)
        elif func == 'cor':
            res = x_.corr()
        else:
            return "wrong func input"
    elif skipmiss == False: #pair_wise
        if func == 'cov':
            res = x.cov()
        elif func == 'cor':
            res = x.corr()
        else:
            return "wrong func input"       
    else:
        return "wrong skipmiss input" 
    
    return res
#1.1
res = missing_cov(x, skipmiss = True, func = 'cov') 
test(res, pd.read_csv(path + 'testout_1.1.csv'),0)
#1.2
missing_cov(x, skipmiss = True, func = 'cor')
#1.3
missing_cov(x, skipmiss = False, func = 'cov')
#1.4
missing_cov(x, skipmiss = False, func = 'cor')


#Test 2 - EW Covariance
x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/test2.csv")
def populateWeights(x, λ):
    n = len(x)
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
populateWeights(x, 0.97)

def EW_calCovariance(x,λ):
    #populate weights
    n = len(x)
    total_weights = 0.0
    w =[]
    for i in range(n):
        w_i = (1-λ) * (λ**(n-1-i))
        total_weights += w_i
        w.append(w_i)
    w = [i/total_weights for i in w]
    c_w = [sum(w[0:i]) for i in range(len(w))]
    weights = pd.DataFrame(w,columns = ['weights'])
    cum_weights = pd.DataFrame(c_w,columns = ["λ="+str(λ)])

    # reduce mean for each obeservation
    reduced_mean= x[x.columns].apply(lambda x : x-np.mean(x))
    i_num = len(x.columns) 
    cov_mat = np.zeros((i_num,i_num))
    for i in range(i_num):
        for j in range(i_num):
            cov_ij = np.dot(np.array(w),reduced_mean[reduced_mean.columns[i]] * reduced_mean[reduced_mean.columns[j]])
            cov_mat[i][j] = cov_ij
    
    return cov_mat
EW_calCovariance(x, 0.97)
#2.2 EW Correlation λ=0.94
def EW_calCorre(cov_matrix):
    sd = 1 / np.sqrt(np.diag(cov_matrix))  # Inverse of the square root of the diagonal elements
    cout = np.diag(sd) @ cov_matrix @ np.diag(sd) 
    return cout
EW_calCorre(EW_calCovariance(x, 0.94))
#2.3 EW Cov w/ EW Var(λ=0.97) EW Correlation(λ=0.94)
cout1 = EW_calCovariance(x, 0.97)
sd1 = np.sqrt(np.diag(cout1))
cout2 = EW_calCovariance(x, 0.94)
sd = 1 / np.sqrt(np.diag(cout2))
cout = np.diag(sd1) @ np.diag(sd) @ cout2 @ np.diag(sd) @ np.diag(sd1)
res = pd.DataFrame(cout, columns = x.columns)

res.to_csv('/Users/ruoqiyan/Documents/545_Assignment_Repository/Week05/res/2.2res.csv',index = False)

#2. Non-PSD fixes for correlation matrices
#Test 3 - non-psd matrices
def near_psd(a, epsilon=0.0):
    n = a.shape[0]
    invSD = None
    out = a.copy()

    # Calculate the correlation matrix if we got a covariance matrix
    if not np.allclose(np.diag(out), 1):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD

    # SVD, update the eigenvalues and scale
    vals, vecs = eig(out)
    vals = np.maximum(vals, epsilon)
    T = 1.0 / (vecs ** 2 @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    # Add back the variance if invSD was calculated
    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD

    return out
#3.1 near_psd covariance
x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/testout_1.3.csv")
near_psd(x)
#3.2 near_psd Correlation
x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/testout_1.4.csv")
near_psd(x)

#3.3 Higham covariance
import numpy as np
from scipy.linalg import sqrtm

def _getAplus(A):
    eigval, eigvec = np.linalg.eigh(A)
    Q = np.diag(np.maximum(eigval, 0))
    return eigvec @ Q @ eigvec.T

def _getPS(A, W):
    W05 = sqrtm(W)
    iW = inv(W05)
    return iW @ _getAplus(W05 @ A @ W05) @ iW

def _getPu(A, W):
    Aret = np.copy(A)
    np.fill_diagonal(Aret, 1)
    return Aret

def wgtNorm(A, W):
    W05 = sqrtm(W)
    return np.sum((W05 @ A @ W05) ** 2)

def higham_nearestPSD(pc, method, epsilon=1e-9, maxIter=100, tol=1e-9):
    n = pc.shape[0]
    W = np.diag(np.ones(n))
    deltaS = 0
    invSD = None

    Yk = np.copy(pc)
    
    # calculate the correlation matrix if we got a covariance
    if not np.allclose(np.diag(Yk), 1.0):
        invSD = np.diag(1.0 / np.sqrt(np.diag(Yk)))
        Yk = invSD @ Yk @ invSD
    if method == 'correlation':
        result = Yk
    
    elif method =='covariance':     
        Yo = np.copy(Yk)
        
        norml = np.finfo(np.float64).max
        i = 1

        while i <= maxIter:
            Rk = Yk - deltaS
            Xk = _getPS(Rk, W)
            deltaS = Xk - Rk
            Yk = _getPu(Xk, W)
            norm = wgtNorm(Yk - Yo, W)
            minEigVal = np.min(np.linalg.eigvalsh(Yk))

            if norm - norml < tol and minEigVal > -epsilon:
                break
            
            norml = norm
            i += 1

        if i < maxIter:
            print("Converged in {} iterations.".format(i))
        else:
            print("Convergence failed after {} iterations".format(i - 1))

        # Add back the variance
        if invSD is not None:
            invSD = np.diag(1.0 / np.diag(invSD))
            Yk = invSD @ Yk @ invSD
        result = Yk
    # set to dataframe 
    return Yk 



#3.3 Higham covariance
x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/testout_1.3.csv")
higham_nearestPSD(x, 'covariance')
#3.4 Higham Correlation
x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/testout_1.4.csv")
higham_nearestPSD(x,'correlation')

#Test 4 - cholesky factorization
def chol_psd(root, a):

    n = a.shape[0]
    root.fill(0.0)  # Initialize the root matrix with 0 values

    for j in range(n):
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])
        
        temp = a.iloc[j][j] - s
        if temp < 1e-8:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        if root[j, j] != 0.0:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a.iloc[i][j] - s) * ir

    return root

x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/testout_3.1.csv")
n,m = x.shape
root = np.zeros((n,m))
chol_psd(root, x)


#3. Simulation Methods
#Test 5 - Normal Simulation
def simulate_normal(N, cov, mean=None, seed=1234, fixMethod = None):
    n, m = cov.shape
    if n != m:
        raise ValueError(f"Covariance matrix is not square ({n},{m})")
    
    if mean is None:
        mean = np.zeros(n)
    else:
        mean = np.array(mean)
        if mean.size != n:
            raise ValueError(f"Mean ({mean.size}) is not the size of cov ({n},{n})")
    
    l = cov
    # Take the root of the covariance matrix
    if fixMethod == 'near_psd':
        l = near_psd(l, epsilon=0.0)
        l = pd.DataFrame(l)
    elif fixMethod == 'higham_nearestPSD':
        l = higham_nearestPSD(l, 'covariance',epsilon=0.0)
        l = pd.DataFrame(l)


    root = np.zeros((n,n))
    L = chol_psd(root, l)
    # Set the seed
    npr.seed(seed)
    
    # Generate random standard normals
    rand_normals = norm.rvs(size=(N, n))

    # Apply the standard normals to the Cholesky root
    out = np.dot(rand_normals, L.T)
    # Add the mean
    out += mean
    
    return out

#5.1 PD Input
x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/test5_1.csv")
res = pd.DataFrame(simulate_normal(100000, x)).cov()
# 5.2 PSD Input
x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/test5_2.csv")
res = pd.DataFrame(simulate_normal(100000, x)).cov()
# 5.3 nonPSD Input, near_psd fix
x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/test5_3.csv")
res = pd.DataFrame(simulate_normal(100000, x, fixMethod = 'near_psd')).cov()
# 5.4 nonPSD Input Higham Fix
x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/test5_3.csv")
res = pd.DataFrame(simulate_normal(100000, x, fixMethod = 'higham_nearestPSD')).cov()


# 5.5 PSD Input - PCA Simulation
def simulate_pca(a, nsim, pctExp=1, mean=None, seed=1234):
    n = a.shape[0]

    # If the mean is missing then set to 0, otherwise use provided mean
    if mean is None:
        _mean = np.zeros(n)
    else:
        _mean = np.array(mean)
    
    # Eigenvalue decomposition
    vals, vecs = np.linalg.eigh(a)
    # Python's eigh returns eigenvalues in ascending order, no need to flip them
    
    # Only keep the positive eigenvalues (and corresponding eigenvectors)
    posv = vals > 1e-8
    vals = vals[posv]
    vecs = vecs[:, posv]
    
    tv = np.sum(vals)

    nval = n
    if pctExp < 1:
        cum_var_explained = np.cumsum(vals) / tv
        nval = np.argmax(cum_var_explained >= pctExp) + 1
        vals = vals[:nval]
        vecs = vecs[:, :nval]
    
    # Calculate the matrix B
    B = vecs @ np.diag(np.sqrt(vals))

    # Set the seed
    npr.seed(seed)
    
    # Generate random normals
    r = npr.randn(nval, nsim)
    
    # Simulate data
    out = (B @ r).T + _mean.reshape(1, -1)
    
    return out
x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/test5_2.csv")
res = simulate_pca(x,100000,pctExp=.99)
res = pd.DataFrame(res).cov()



#4. VaR calculation methods
# Test 6
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
    
    return df3[1:]
# 6.1 Arithmetic returns
x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/test6.csv")
res = return_calculate(x)
# 6.2 Log returns
x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/test6.csv")
res = return_calculate(x,method = "LOG")

# Test 7
class FittedModel:
    def __init__(self, stat, beta, error_model, eval_fn, errors, u):
        self.stat = stat
        self.beta = beta
        self.error_model = error_model
        self.eval = eval_fn
        self.errors = errors
        self.u = u
# Test 8
from scipy.stats import norm
from scipy.integrate import quad

def VaR(a, alpha=0.05):
    """Calculate the Value at Risk (VaR) for a dataset."""
    x = np.sort(a)
    n = len(a)
    nup = int(np.ceil(n * alpha))
    ndn = int(np.floor(n * alpha))
    v = 0.5 * (x[nup - 1] + x[ndn - 1])  # Adjusted indexing for Python (0-based)
    return -v

def VaR_distribution(d, alpha=0.05):
    """Calculate the Value at Risk (VaR) for a distribution."""
    return -d.ppf(alpha)

def ES(a, alpha=0.05):
    """Calculate the Expected Shortfall (ES) for a dataset."""
    x = np.sort(a)
    n = len(a)

    nup = int(np.ceil(n * alpha))
    ndn = int(np.floor(n * alpha))
    v = 0.5 * (x[nup - 1] + x[ndn - 1])  # Adjusted indexing for Python (0-based)
    es = np.mean(x[x <= v])
    return -es


def ES_distribution(d, alpha=0.05):
    """Calculate the Expected Shortfall (ES) for a distribution."""
    v = VaR_distribution(d, alpha=alpha)
    
    def integrand(x):
        return x * d.pdf(x)
    
    st, _ = quad(integrand, d.ppf(1e-12), -v) #quad is the function used for integration.
    return -st / alpha


# 7.1 Fit Normal Distribution
def fit_normal(x):
    # Calculate mean and standard deviation
    m = np.mean(x)
    s = np.std(x, ddof=1)  # Using Bessel's correction
    
    # Define the error model based on the normal distribution
    error_model = norm(loc=m, scale=s)
    
    # Calculate the errors and their cumulative probabilities (u values)
    errors = x - m
    u = error_model.cdf(x)
    
    # Define a function to evaluate model quantiles at given u values
    eval_fn = lambda u: error_model.ppf(u)
    
    # Construct and return a FittedModel instance
    return FittedModel(None, None, error_model, eval_fn, errors, u)


x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/test7_1.csv")
model = fit_normal(x)
model.error_model.mean()# 0.04602574
model.error_model.std() # 0.04677994
# Test 8.1 VaR Normal
print("VaR Absolute", VaR_distribution(model.error_model)) # 0.03092042
print("VaR Diff from Mean",-VaR_distribution(model.error_model)- model.error_model.mean()) # 0.07694615 -norm.ppf(0.05, loc=0, scale=model.error_model.std())
# Test 8.4 ES Normal
print("ES Absolute", ES_distribution(model.error_model))  # 0.05046784401426306
print("ES Diff from Mean", -ES_distribution(model.error_model)- model.error_model.mean()) #-0.09649358



# 7.2 Fit TDist
from scipy.stats import t 
from scipy.optimize import minimize
from scipy.stats import kurtosis

def general_t_ll(mu, s, nu, x):
    rv = t(df=nu, loc=mu, scale=s)
    return np.sum(np.log(rv.pdf(x)))

def fit_general_t(x):
    # Initial guesses based on moments
    nu,m,s  = t.fit(x)

    # Create the error model
    error_model = t(df=nu, loc=m, scale=s)
    
    # Calculate errors and U
    errors = x - m
    u = error_model.cdf(x)

    # Evaluation function for quantiles
    eval_fn = lambda u: error_model.ppf(u)

    return FittedModel([m, s, nu], None, error_model, eval_fn, errors, u)

x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/test7_2.csv")
model = fit_general_t(x['x1'])
model.error_model.mean() # 0.04618911185061204
#model.error_model.std() # 0.05376800450032772 ???
model.stat # [0.04594038004735414, 0.04544287220830122, 6.336866997308613]
# Test 8.2 VaR TDist
print("VaR Absolute", VaR_distribution(model.error_model)) # 0.041529702716233574
print("VaR Diff from Mean", VaR_distribution(model.error_model)+ model.error_model.mean()) # 0.08747008276358771
# Test 8.5 ES TDist
print("ES Absolute", ES_distribution(model.error_model)) # 0.07523208716011755
print("ES Diff from Mean", -ES_distribution(model.error_model)- model.error_model.mean()) # 0.12117246720747168


# 7.3 Fit T Regression
from scipy.optimize import minimize
from scipy import stats
from numpy.linalg import inv
from scipy.stats import t

def fit_regression_t(y, X):
    # Add intercept
    X = np.hstack((np.ones((len(X), 1)), X))
    n, nB = X.shape

    # Initial estimates based on OLS
    b_start = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ b_start
    start_m = residuals.mean()
    start_s = residuals.std(ddof=2)
    start_nu = 6.0 / stats.kurtosis(residuals, fisher=False) + 4

    # Define optimization objective
    def objective(params):
        mu, s, nu = params[:3]
        beta = params[3:]
        return -general_t_ll(mu, s, nu, y - X @ beta)
    
    # Initial parameter guess
    initial_guess = [start_m, start_s, start_nu] + b_start.tolist()

    # Optimization
    result = minimize(objective, initial_guess, method='L-BFGS-B')

    # Extract optimized parameters
    m, s, nu = result.x[:3]
    beta = result.x[3:]

    # Constructing FittedModel
    error_model = t(df=nu, loc=m, scale=s)

    def eval_fn(x, u = 0.5):
        return t.ppf(u, df=nu, loc=m, scale=s)

    errors = y - eval_fn(X, 0.5)
    u = t.cdf(errors, df=nu, loc=m, scale=s)

    return FittedModel([m, s, nu],beta, error_model, eval_fn, errors, u)

x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/test7_3.csv")
model = fit_regression_t(x['y'],x[['x1', 'x2', 'x3']])
print('mu: ', model.error_model.mean()) # 8.743006318923108e-18
print('sigma: ', model.error_model.std()) # 0.06311884573136509
print('nu: ', model.error_model) # 5.396697627453499
print('Alpha,beta: ', model.beta) #[0.04156686 1.02843062 2.18469665 3.17289129]
model.stat #[8.743006318923108e-18, 0.06311884573136509, 5.396697627453499]
# Test 8.3 VaR Simulation
model = fit_general_t(x)
sim = model.eval(np.random.rand(10000))
print("VaR Absolute", VaR(sim)) # 0.040364295532057046
print("VaR Diff from Mean",VaR(sim)+np.mean(sim)) #0.0868257807422640
# Test 8.6 VaR Simulation
print("ES Absolute", ES(sim)) # 0.07329953615927434
print("ES Diff from Mean",ES(sim)+np.mean(sim)) # 0.11976102136948133


#5. ES calculation
#Calculate arithmetic returns
# Test 9
# 9.1
x = pd.read_csv("/Users/ruoqiyan/Documents/545_Assignment_Repository/testfiles/data/test9_1_returns.csv")

from scipy.stats import norm, spearmanr

cin_A = x['A']
cin_B = x['B']

prices = {"A": 20.0, "B": 30.0}

# Model fitting
models = {
    "A": fit_normal(cin_A),  # This function needs to be defined or adapted
    "B": fit_general_t(cin_B)  # This function needs to be defined or adapted
}

nSim = 100000

U = np.column_stack((models["A"].u, models["B"].u))

# Spearman correlation
spcor, _r = spearmanr(U)
cov = np.array([[1,spcor],[spcor, 1]])


# PCA simulation (placeholder for your simulate_pca function)
uSim = simulate_pca(cov,nSim)

# Convert simulated values to quantiles using the CDF of a standard normal distribution
uSim = norm.cdf(uSim)

# Simulated returns (you'll need to adapt models["A"].eval and models["B"].eval to Python)
simRet = pd.DataFrame({
    "A": models["A"].eval(uSim[:, 0]),  # This method needs to be defined or adapted
    "B": models["B"].eval(uSim[:, 1])  # This method needs to be defined or adapted
})

portfolio = pd.DataFrame({
    "Stock": ["A", "B"],
    "currentValue": [2000.0, 3000.0]
})

# Cross join not directly available in pandas, use a workaround
iteration = range(1, nSim + 1)
values = pd.merge(portfolio.assign(key=1), pd.DataFrame({"iteration": iteration, "key": np.ones(len(iteration))}), on="key")
values.drop("key", inplace = True, axis = 1)


# Calculate PnL and simulated value
simulatedValue = np.zeros(len(values))
pnl = np.zeros(len(values))
for i in range(len(values)):
    currentValue = values.iloc[i]['currentValue']
    iter = values.iloc[i]['iteration'] - 1  # Adjust for zero-based index
    stock = values.iloc[i]['Stock']
    simulatedValue[i] = currentValue * (1 + simRet.loc[iter, stock])
    pnl[i] = simulatedValue[i] - currentValue


values['pnl'] = pnl
values['simulatedValue'] = simulatedValue

var = []
es = []
var_pct = []
es_pct = []

stock_A = values[values['Stock'] == "A"]
A_current = stock_A['simulatedValue'].iloc[-1]
x = stock_A['pnl']
a_var = VaR(x) #94.579514546366
var.append(a_var)
a_es = ES(x) #118.53735394382251
es.append(a_es)
a_var_pct = VaR(x)/A_current #0.046144255840017426
var_pct.append(a_var_pct)
a_ex_pct = ES(x)/A_current #0.0578330097507633
es_pct.append(a_ex_pct)

stock_B = values[values['Stock'] == "B"]
B_current = stock_B['simulatedValue'].iloc[-1]
x = stock_B['pnl']
b_var =  VaR(x) #108.08952714642146
var.append(b_var)
b_es = ES(x) #152.34696624753286
es.append(b_es)
b_var_pct = VaR(x)/B_current #0.035089288387111284
var_pct.append(b_var_pct)
b_ex_pct = ES(x)/B_current #0.04945665666868608
es_pct.append(b_ex_pct)
 


# TOTAL
values['currentValue'] = currentValue
values['simulatedValue'] = simulatedValue
values['pnl'] = pnl


# Total Metrics
gdf = values.groupby('iteration')
# aggregate to totals per simulation iteration
totalValues = gdf.agg(
    currentValue=('currentValue', 'sum'),
    simulatedValue=('simulatedValue', 'sum'),
    pnl=('pnl', 'sum')
).reset_index()

x = totalValues['pnl']
current = totalValues['currentValue'].iloc[0]
ttl_var = VaR(x) #153.9002522009032
var.append(ttl_var)
ttl_es = ES(x) # 202.22401640388208
es.append(ttl_es)
ttl_var_pct = VaR(x)/current #0.030780050440180638
var_pct.append(ttl_var_pct)
ttl_es_pct = ES(x)/current #0.04044480328077642
es_pct.append(ttl_es_pct)

# Final Output
riskOut = pd.DataFrame()
riskOut['Stock'] = ['A', 'B', 'Total']
riskOut['VaR95'] = var
riskOut['ES95'] = es
riskOut['VaR95_Pct'] = var_pct
riskOut['ES95_Pct'] = es_pct
riskOut.set_index('Stock',inplace = True)


