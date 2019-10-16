import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scipy

"""
CONSTANTS
"""
min=-1 # min x value
max=1 # max x value
n=100 # number of samples
step=5
w=np.matrix([[1.3,.5]]) # True value
w0=np.matrix([[0],[0]]) # Prior mu
s0=np.eye(2) # Prior variance

"""
Content
"""
# Generate y values for sample of Xs
def y_i(X_i):
    e=np.random.normal(0,0.3,X_i.shape[0]) # additive Noise
    return (w*X_i.T+e).T

# Plots distribution of posterior as contour plot
def plot_distribution(ax,mu,Sigma,alpha):
    x = np.linspace(-1.5,1.5,100)
    x1p, x2p = np.meshgrid(x,x)
    pos = np.vstack((x1p.flatten(), x2p.flatten())).T
    pdf = scipy.multivariate_normal(mu.flatten(), Sigma)
    Z = pdf.pdf(pos)
    Z = Z.reshape(100,100)
    ax.contour(x1p,x2p,Z, 5, colors='r', lw=5, alpha=alpha)
    ax.set_xlabel('w_0')
    ax.set_ylabel('w_1')
    return

# Calculate mean & variance of posterior
def calculate_posterior(X,y,s0,w0,beta):
    # mu
    s0_inv=np.linalg.inv(s0)
    mu_1=np.linalg.inv(s0_inv+beta*X.T.dot(X))
    mu_2=(s0_inv*w0+beta*X.T*y)
    mu=mu_1*mu_2

    # sigma^2
    sigma2=np.linalg.inv(s0_inv+beta*X.T.dot(X))

    return mu, sigma2

fig,axs=plt.subplots(nrows=1,ncols=2)

# Add elements of plots which will not change during iteration
axs[0].set_xlabel("x")
axs[0].set_xlabel("y")
axs[0].plot([0],[0],c="r",alpha=0) # Predicted
axs[0].plot([-1,1],[-w[0,0]+w[0,1],w[0,0]+w[0,1]],c="g") # True line
axs[0].legend(("Predicted", "True"))

# True values
axs[1].scatter(x=w[0,0],y=w[0,1],marker="x",c="g",s=100)

# Increase the number of samples & plot posterior
for i in range(step,n,step):
    # Generate X
    values=np.linspace(min,max,num=i)
    X=np.zeros((i,2))
    for j in range(0,i):
        X[j,0]=values[j]
        X[j,1]=1
    y=y_i(X)

    # Calculate posterior
    mu,sigma2=calculate_posterior(X,y,s0,w0,.3)

    # Plot X values against y
    axs[0].scatter(y=np.asarray(y[:,0]),x=np.asarray(X[:,0]))

    # Plot distribution
    plot_distribution(axs[1],mu.A,sigma2,alpha=i**2/n**2)

    # Animation over a total of 10 seconds
    plt.pause((10*step)/n)
plt.show()
