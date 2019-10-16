import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

dist_from_mean=[]

def posterior(a,b,X):
    a_n=a+X.sum()
    b_n=b+(X.shape[0]-X.sum())

    return beta.pdf(mu_test,a_n,b_n)

# Parameters for generating data
mu=.2 # mu^*
N=200

# Generate the data
X=np.random.binomial(1,mu,N)
mu_test=np.linspace(0,1,100)

# Initial estimates
a=10
b=10
prior_mu=beta.pdf(mu_test,a,b)

prior_mean=np.mean(prior_mu)

# create figure
fig,ax= plt.subplots(nrows=1,ncols=2)

# plot prior
ax[0].plot(mu_test,prior_mu,'g') # Plot initial estimate
ax[0].fill_between(mu_test,prior_mu,color='green',alpha=0.3)
ax[0].set_xlabel('$\mu$')
ax[0].set_ylabel('$p(\mu|\mathbf{x})$')

# lets pick a random (uniform) point from the data
# and update our assumption with this
index = np.random.permutation(X.shape[0])

# Iterate through data, creating a new posterior with one more data point each time
for i in range(0,X.shape[0]):
    y = posterior(a,b,X[:index[i]])
    ax[0].plot(mu_test,y,'r',alpha=1-i/N)
    dist_from_mean.append(prior_mean-np.mean(y))

# Plot final posterior
y = posterior(a,b,X)
ax[0].plot(mu_test,y,'b',linewidth=4.0)
ax[1].plot(dist_from_mean)
plt.show()
