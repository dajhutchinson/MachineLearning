import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def add_gaussian_noise(im,prop,varSigma):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    e = varSigma*np.random.randn(np.prod(im.shape)).reshape(im.shape)
    im2 = np.copy(im).astype('float')
    im2[index] += e[index]
    return im2

def add_saltnpeppar_noise(im,prop):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    im2 = np.copy(im)
    im2[index] = 1-im2[index]
    return im2

def neighbours(i,j,M,N,size=4):
    if size==4:
        if (i==0 and j==0): n=[(0,1), (1,0)]
        elif i==0 and j==N-1: n=[(0,N-2), (1,N-1)]
        elif i==M-1 and j==0: n=[(M-1,1), (M-2,0)]
        elif i==M-1 and j==N-1: n=[(M-1,N-2), (M-2,N-1)]
        elif i==0: n=[(0,j-1), (0,j+1), (1,j)]
        elif i==M-1: n=[(M-1,j-1), (M-1,j+1), (M-2,j)]
        elif j==0: n=[(i-1,0), (i+1,0), (i,1)]
        elif j==N-1: n=[(i-1,N-1), (i+1,N-1), (i,N-2)]
        else: n=[(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
        return n
    if size==8:
        print('Not yet implemented\n')
        return -1

def ICM(img,t=1):
    # THIS IS ALL BULLSHITE
    # DOES NOT WORK
    vals=[-1,1]
    x=[vals[np.random.binomial(1,.5)] for _ in range(len(img)*len(img[0]))]
    x=np.array(x).reshape(len(img),len(img[0]))
    print(x.shape)
    for _ in range(t):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                neighs=[img[i][3] for i in neighbours(i,j,x.shape[0],x.shape[1])]
                x[i,j]=scipy.stats.mode(neighs)[0]
    return x

# proportion of pixels to alter
prop = 0.7
varSigma = 0.1

im = plt.imread('dog.png')
fig = plt.figure()
ax = fig.add_subplot(141)
ax.imshow(im,cmap='gray')

im2 = add_gaussian_noise(im,prop,varSigma)
ax2 = fig.add_subplot(142)
ax2.imshow(im2,cmap='gray')

im2 = add_saltnpeppar_noise(im,prop)
ax3 = fig.add_subplot(143)
ax3.imshow(im2,cmap='gray')

im3=ICM(im2)
ax4=fig.add_subplot(144)
ax4.imshow(im3,cmap="gray")

plt.show()
