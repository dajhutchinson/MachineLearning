import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import norm

def rbf_kernel(x_i,x_j,ell2,sigma2):
    if x_j is None:
        d = cdist(x_i, x_i)
    else:
        d = cdist(x_i, x_j)
    K = sigma2*np.exp(-np.power(d, 2)/ell2)
    return K

def gp_prediction(x1, y1, xstar, lengthScale, varSigma):
    k_starx = rbf_kernel(xstar,x1,lengthScale,varSigma)
    k_xx = rbf_kernel(x1, None, lengthScale, varSigma)
    k_starstar = rbf_kernel(xstar,None,lengthScale,varSigma)

    mu = k_starx.dot(np.linalg.inv(k_xx)).dot(y1)
    var = k_starstar - (k_starx).dot(np.linalg.inv(k_xx)).dot(k_starx.T)

    return mu, var, xstar

def surrogate_belief(x,f,x_star,theta):
    mu_star,varSigma_star,_=gp_prediction(x,f,x_star,1,theta)
    return mu_star,varSigma_star

# x is a vector of points to explore
# mu, varSigma are the mean & variance at these points
def expected_improvement(f_,mu,varSigma,x):
    alpha=(f_-mu)*norm.cdf(x,loc=mu,scale=varSigma)
    alpha+=varSigma*norm.pdf(f_,loc=mu,scale=varSigma)
    return alpha

# experiments
def f(x, beta=0, alpha1=1.0, alpha2=1.0):
    return np.sin(3.0*x) - alpha1*x + alpha2*x**2 + beta*np.random.randn(x.shape[0])
