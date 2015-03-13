#!/usr/bin/env python3

"""
c4_2.py

This solves HW #4 coding question #2
"""

from scipy.io import loadmat
import os.path
from PROJECT import project 
from kernelSVM import kernel_classify
import numpy

# when refactoring, just put this in a function that returns X,y 
DATADIR = './data/'
matfile = os.path.join(DATADIR, 'MNIST_SMALL.mat')

def _make_gaussian_kernel(X, sigma):
    """
    returns the gaussian kernel, i.e. kernel with
    M(x,y) := exp{ - ||x-y||^2 / (2σ^2) }

    returns the matrix M with M_(i,j) = M(x_i, x_j) 

    ... using linear algebra tricks, not sure if this is really better
    """
    
    temp = [row.T.dot(row) for row in X]

    # this is how you make a column vector from a list???
    Q = numpy.array(temp, ndmin=2).T
    Q = numpy.tile(Q, (1, img.shape[0]))
    xxt = X.dot(X.T)

    M = Q - xxt

    # now M contains ||x_i - x_j||^2

    M = M + M.T
    M = -M / (2*sigma**2)

    return numpy.exp(M)


     

d = loadmat(matfile)
img, lab = d['img'], d['lab']

# use σ=5 because that's what the doctor said
M = _make_gaussian_kernel(img, 5)

# y contains class labels with y_i = {  1   x_i corresponds to a 9
#                                    { -1   else
y = (lab == 9).astype('double')
y = 2*y - 1 

C = 7.5
gamma = kernel_classify(M, y, C)

# classifying step
supp = (0 < abs(gamma)) & (abs(gamma) < C)

s = numpy.nonzero(supp)[0]

alpha = (y[s] - X[s].dot(beta)).sum() / s.size

