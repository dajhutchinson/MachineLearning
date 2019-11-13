import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x=np.linspace(-6,6,200)

# 3 possible distributions
pdf1=norm.pdf(x,0,1)
pdf2=norm.pdf(x,1,3)
pdf3=norm.pdf(x,-2.5,.5)

# Average across the distributions (With likelihood of each distribution)
pdf4=(.3*pdf1)+(.2*pdf2)+(.5*pdf3)

fig,ax=plt.subplots(nrows=1,ncols=1)

# Plot Distributions
ax.plot(x,pdf1,color="r",alpha=.5)
ax.fill_between(x,pdf1,color='r',alpha=.3)
ax.plot(x,pdf2,color="g",alpha=.5)
ax.fill_between(x,pdf2,color='g',alpha=.3)
ax.plot(x,pdf3,color="b",alpha=.5)
ax.fill_between(x,pdf3,color='b',alpha=.3)
ax.plot(x,pdf4,color="k",alpha=.5,linewidth=3,linestyle="--") # gray
ax.fill_between(x,pdf4,color='k',alpha=.3)

plt.show()
