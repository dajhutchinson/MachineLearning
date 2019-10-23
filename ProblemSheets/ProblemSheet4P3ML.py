import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# co-variance function
# ell = length scale
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

N=5
x1=np.linspace(-3.1  ,3,N).reshape(-1,1)
y1=np.sin(2*np.pi/x1) + x1*.1 +.3*np.random.randn(x1.shape[0]).reshape(-1,1)

x_star=np.linspace(-6,6,200).reshape(-1,1)

Nsamp=100
mu_star,var_star,x_star=gp_prediction(x1,y1,x_star,2,1)
f_star=np.random.multivariate_normal(mu_star.reshape(-1,),var_star,Nsamp)

f_mean=np.array([np.mean(f_star[:,i]) for i in range(0,f_star.shape[1])])
f_sd=np.array([np.std(f_star[:,i]) for i in range(0,f_star.shape[1])])

fig,ax=plt.subplots(nrows=1,ncols=2)
ax[0].plot(x_star,f_star.T,zorder=0)
ax[0].scatter(x1,y1,200,'k','*',zorder=1)

ax[1].plot(x_star,f_mean,c="blue",zorder=0)
ax[1].plot(x_star,f_mean-f_sd,c="blue",linestyle="--",zorder=0)
ax[1].plot(x_star,f_mean+f_sd,c="blue",linestyle="--",zorder=0)
ax[1].fill_between(x_star.reshape(-1,),f_mean+f_sd,f_mean-f_sd,alpha=.3,zorder=0)
ax[1].scatter(x1,y1,200,'k','*',zorder=1)

plt.show()
