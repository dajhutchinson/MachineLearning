import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scipy

def plot_line(ax,w):
    # Consider points 5 & -5
    X=np.zeros((2,2))
    X[0,0]=-5
    X[1,0]=5
    X[:,1]=1

    # Because of the concatination we have to flip the transpose
    y=w.dot(X.T)
    print(X[:,0])
    exit()
    ax.plot(X[:,0],y)

# Create prior distribution
tau=1.0*np.eye(2) # Prior variance
w_0=np.zeros((2,1)) # Prior mean

# Sample from prior
n_samples=100
w_sample=np.random.multivariate_normal(w_0.flatten(), tau, size=n_samples)

# create plot
fig,ax=plt.subplots(nrows=1,ncols=1)

for i in range(0,w_sample.shape[0]):
    plot_line(ax,w_sample[i,:])

plt.show()

"""
1. Most likely is w=(0,0) since most lines are approximately horizontal & run through (0,0)
2. w=(infty,infty)
3. No, some have very small likelihood but non are zero since normal distribution is used
"""

"""
1. Asuume beta=0 => N(w|mu=w0,sigma^2=S0)
2. Assume w0=[0,...,0] => N(w|(S0^-1+betaX^TX)^-1(beta X^Ty), (S0^-1+beta X^TX)^-1)
3. Assume y=[], X=[] => N(w|mu=w0,sigma^2=S0)
4. As X & y increase in size,  they will dominate the expressions since X^TX produces square terms which are always non-negative
"""
