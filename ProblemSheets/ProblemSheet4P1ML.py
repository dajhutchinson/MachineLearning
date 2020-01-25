import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

"""
    DIFFERENT KERNELS
    -----------------
    These are used for the covariance function in Gaussian Processes

    +-------+----------+
    | RBF   | Linear   |
    +-------+----------+
    | White | Periodic |
    +-------+----------+
"""
# mean function
def mu(x_i):
    return np.zeros(x_i.shape)

# co-variance function
# ell = length scale
def rbf_kernel(x_i,x_j,sigma2,ell2):
    if x_j is None:
        d = cdist(x_i, x_i)
    else:
        d = cdist(x_i, x_j)
    K = sigma2*np.exp(-np.power(d, 2)/ell2)
    return K

def lin_kernel(x_i,x_j,sigma2):
    if x_j is None:
        return sigma2*x_i.dot(x_i.T)
    else:
        return sigma2*x_i.dot(x_j.T)

def white_kernel(x_i,x_j,sigma2):
    if x_j is None:
        return sigma2*np.eye(x_i.shape[0])
    else:
        return sigma2*np.eye(x_i.shape[0],x_j.shape[0])

def periodic_kernel(x_i,x_j,sigma2,period,ell):
    if x_j is None:
        d = cdist(x_i, x_i)
    else:
        d = cdist(x_i, x_j)
    return sigma2*np.exp(-(2*np.sin((np.pi/period)*np.sqrt(d))**2)/ell**2)

# index set for the marginal
x=np.linspace(-6,6,200).reshape(-1,1) # (200,1) rather than (200,)
# computer covariance matrix
K_1=rbf_kernel(x,None,1,2)
K_2=lin_kernel(x,None,1)
K_3=white_kernel(x,None,1)
K_4=periodic_kernel(x,None,1,1,2)
# create mean vector
mu=np.zeros(x.shape[0])

# Plot samples
fig,ax=plt.subplots(nrows=2,ncols=2)
ax[0,0].plot(x,np.random.multivariate_normal(mu, K_1, 10).T)
ax[0,1].plot(x,np.random.multivariate_normal(mu, K_2, 10).T)
ax[1,0].plot(x,np.random.multivariate_normal(mu, K_3, 10).T)
ax[1,1].plot(x,np.random.multivariate_normal(mu, K_4, 10).T)
plt.show()
