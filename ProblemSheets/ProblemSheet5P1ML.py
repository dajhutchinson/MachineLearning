"""
Unsupervised learning
Principal Component Analaysis
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Maximum Likelihood of W (weightings)
# eigenvectors for PCA
# eigenvectors of covariance matrix of Y (Y=Training data) with q largest eigenvalues
def MLW(Y,q):
    v,W=np.linalg.eig(np.cov(Y.T)) # eigenvalues & vectors of covariance matrix of training data
    idx=np.argsort(np.real(v))[::-1][:q] # q largest eigenvalues ([::-1] reverses items in array)
    return np.real(W[:,idx])

# posterior distribution of latent variable
# PCA
def posterior(w,x,mu_x,beta):
    A=np.linalg.inv(w.dot(w.T)+(1/beta)*np.eye(w.shape[0])) # (ww^T+1/b*I)-1
    mu=w.T.dot(A.dot(x-mu_x)) #w^T(ww^T+1/b*I)-1(x-mu_x)
    varSigma=np.eye(w.shape[1])-w.T.dot(A.dot(w)) # I - W^T(ww^T+1/b*I)-1W

    return mu, varSigma

# Generate spiral (True distribution)
t=np.linspace(0,3*np.pi,100) # 100 equal intervals in [0,3pi]
x=np.zeros((t.shape[0],2))
x[:,0]=t*np.sin(t)
x[:,1]=t*np.cos(t)

# Generate training data (ie random linear transformation of true distribution)
# pick a random matrix that maps to Y
w=np.random.randn(10,2) # 10x2 random samples from N(0,1)
y=x.dot(w.T) # Xw^T (100,10)
y+=np.random.randn(y.shape[0],y.shape[1]) # random noise
mu_y=np.mean(y,axis=0)

# get maximum likelihood solution of W
w=MLW(y,2)

# compute predictions for latent space, X
xpred=np.zeros(x.shape)
varSigma=[]
for i in range(0,y.shape[0]):
    xpred[i,:],varSigma=posterior(w,y[i,:],mu_y,1/2)

# Plot true X & predicted X
"""plt.plot(x[:,0],x[:,1])
plt.plot(xpred[:,0],xpred[:,1])
plt.show()"""

# Generate Density
N=300
x1=np.linspace(np.min(xpred[:,0]),np.max(xpred[:,0]),N)
x2=np.linspace(np.min(xpred[:,1]),np.max(xpred[:,1]),N)
x1p,x2p=np.meshgrid(x1,x2) # Creates NxN mesh
pos=np.vstack((x1p.flatten(), x2p.flatten())).T # Stacks layers

# compute posterior
Z=np.zeros((N,N))
for i in range(0,xpred.shape[0]):
    pdf=multivariate_normal(xpred[i,:].flatten(),varSigma)
    Z+=pdf.pdf(pos).reshape(N,N)

# plot density
fig,ax=plt.subplots(nrows=1,ncols=2)
ax[0].scatter(xpred[:,0],xpred[:,1])
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(Z,cmap="hot")
ax[1].set_ylim(ax[1].get_ylim()[::-1])
ax[1].set_xticks([])
ax[1].set_yticks([])

plt.show()
