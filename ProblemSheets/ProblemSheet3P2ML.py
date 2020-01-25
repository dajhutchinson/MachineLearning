import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scipy

"""
DESCRIPTION - Generates a set of training data.
              Each cycle it increases the amount of training data used to predict a linear relationship between the dependent & independent variables.
              Left = Observed data, True line & predicted line
              Right = Predicted Distribution of parameters
"""

"""
CONSTANTS
"""
min=-1 # min x value
max=1 # max x value
n=100 # number of samples
step=5 # How much sample size is increased by each cycle
w=np.matrix([[-1.3,.5]]) # True parameter values
w_0=np.matrix([[0],[0]]) # Prior for parameter values
S_0=np.eye(2) # Prior variance (Independent with variance=1)

"""
Content
"""
# Generate y values for sample of Xs (with additive noise)
def y_is(X_i):
    e=np.random.normal(0,0.3,X_i.shape[0]) # additive noise
    return (w*X_i.T+e).T

# Plots distribution of posterior as contour plot
def plot_distribution(ax,m_n,Sigma,alpha=1):
    x = np.linspace(-1.5,1.5,100)
    x1p, x2p = np.meshgrid(x,x)
    pos = np.vstack((x1p.flatten(), x2p.flatten())).T
    pdf = scipy.multivariate_normal(m_n.flatten(), Sigma)
    Z = pdf.pdf(pos)
    Z = Z.reshape(100,100)
    ax.contour(x1p,x2p,Z, 5, colors='r', alpha=alpha)
    ax.set_xlabel('w_0')
    ax.set_ylabel('w_1')
    return

# Calculate posterior mean & variance (Least-Squares Estimates)
def calculate_posterior(X,y,S_0,w_0,beta):
    # m_n
    s0_inv=np.linalg.inv(S_0)
    mu_1=np.linalg.inv(s0_inv+beta*X.T.dot(X)) # Formula for general least-squares estimate (applied to mean)
    mu_2=(s0_inv*w_0+beta*X.T*y)
    m_n=mu_1*mu_2

    # S_n
    S_n=np.linalg.inv(s0_inv+beta*X.T.dot(X)) # Formula for general least-squares estimate (applied to variance)

    return m_n, S_n

# Use estimate of posterior to make predictions
def predictive_posterior(m_0,S_0,beta,x_star,X,y):
    m_n,S_n=calculate_posterior(X,y,S_0,w_0,beta) # Calculate least-squares estimate for mean & variance

    m_star=m_n.T.dot(x_star.T) # Predict mean of each point
    S_star=(1/beta)+x_star.dot(S_n).dot(x_star.T) # Predict variance of each point

    return m_star[0,0], S_star[0,0]

# run simulation where size of training set is increased each cycle
def simulation():
    fig,axs=plt.subplots(nrows=1,ncols=2)

    # Add elements of plots which will not change during iteration
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].plot([0],[0],c="r",alpha=0) # Predicted
    axs[0].plot([min,max],[min*w[0,0]+w[0,1],max*w[0,0]+w[0,1]],c="g") # True line
    axs[0].legend(("Predicted", "True"))

    # True values
    axs[1].scatter(x=w[0,0],y=w[0,1],marker="x",c="g",s=100)

    # Generate X
    values=np.linspace(min,max,num=n)
    X=np.zeros((n,2))
    for i in range(0,n):
        X[i,0]=values[i]
        X[i,1]=1 # columns of 1s for constant
    y=y_is(X)

    # Increase the number of samples & plot posterior
    index = np.random.permutation(X.shape[0])
    for i in range(step,n,step):
        X_i=X[index[0:i],:]
        y_i=y[index[0:i]]

        # Calculate posterior
        m_n,S_n=calculate_posterior(X_i,y_i,S_0,w_0,.3)

        # Plot X values against y
        axs[0].scatter(y=np.asarray(y_i[:,0]),x=np.asarray(X_i[:,0]))
        axs[0].plot([min,max],[min*m_n[0,0]+m_n[1,0],max*m_n[0,0]+m_n[1,0]],c="r",alpha=i**2/n**2) # Predicted

        # Plot distribution
        plot_distribution(axs[1],m_n.A,S_n,alpha=i**2/n**2)

        # Animation over a total of 10 seconds
        #plt.pause((10*step)/n)
        plt.pause(.5)
    plt.show()

x=scipy.uniform(loc=min,scale=max-min).rvs(1)
simulation()
