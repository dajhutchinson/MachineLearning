from scipy.stats import beta
import numpy as np
import matplotlib.pyplot as plt
# create random variable
rv = beta(10.0, 10.0)
# create an index set for the distribution
index = np.linspace(0+1.0e-8, 1, 100)
# create plot
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
# plot density function
ax.plot(index, rv.pdf(index), color='g')
ax.fill_between(index, rv.pdf(index), color='g', alpha=0.3)
# plot cumulative density function
ax.plot(index, rv.cdf(index), color='r')
ax.fill_between(index, rv.cdf(index), color='r', alpha=0.3)
# sample from random variable
Y = rv.rvs(1000)
# plot histogram
hist, bins = np.histogram(Y)
ax.bar(bins[:-1], hist/hist.sum(), alpha=0.4, color='b',width=0.05)
plt.show()
