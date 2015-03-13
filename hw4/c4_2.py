#!/usr/bin/env python3

"""
c4_2.py

This solves HW #4 coding question #2
"""

from scipy.io import loadmat
import os.path
from PROJECT import project 
from kernelSVM import kernel_classify
from loadMNIST import load as load_mnist
from SVMclassify import svm_classify
import numpy


# when refactoring, just put this in a function that returns X,y 
DATADIR = './data/'
matfile = os.path.join(DATADIR, 'MNIST_SMALL.mat')

def _make_row_norm_matrix(X):
    """
    if X is an nxd matrix, return an nxn matrix that
    looks like

    [[ ||x1||^2 ||x1||^2 ... ||x1||^2 ]
     [ ||x2||^2 ||x2||^2 ... ||x2||^2 ]
     ...
     [ ||xn||^2 ||xn||^2 ... ||xn||^2 ]]

    where xi represents a row of X.

    this subfunction is useful for making the gaussian kernel and
    testing whats-it matrix N
    """

    temp = [row.T.dot(row) for row in X]

    Q = numpy.array(temp, ndmin=2).T
    Q = numpy.tile(Q, (1, X.shape[0]))

    return Q

def _make_gaussian_kernel(X, sigma):
    """
    returns the gaussian kernel, i.e. kernel with
    M(x,y) := exp{ - ||x-y||^2 / (2σ^2) }

    returns the matrix M with M_(i,j) = M(x_i, x_j) 

    ... using linear algebra tricks, not sure if this is really better
    """
    
    Q = _make_row_norm_matrix(X)

    xxt = X.dot(X.T)

    M = Q - xxt

    # now M contains ||x_i - x_j||^2
    M = M + M.T

    M = -M / (2*sigma**2)

    return numpy.exp(M)

def _make_gaussian_N(Z, X, sigma):
    """
    M(x,y) := exp{ - ||x-y||^2 / (2σ^2) }

    returns the matrix N with N_(i,j) = M(z_i, x_j) 

    where z_i are the testing points to be classified

    """
    QX = _make_row_norm_matrix(X) 
    QZ = _make_row_norm_matrix(Z)
    
    M = QZ + QX.T - 2*Z.T.dot(X)
    M = -M / (2 * sigma**2 )
    
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
v = M.dot(gamma)
alpha = (y[s] - v[s]).sum() / s.size

img_t, lab_t = load_mnist([4,7,9], 'test')
binary = numpy.array(lab_t, ndmin=2).T

#N = _make_gaussian_N(img_t, img, sigma=5)

# make N ... i'm resorting to list comp because the dimensions are bad ...
sigma = 5
N =  [[ (zi - xj).T.dot(zi - xj) / (2 * sigma**2) for xj in img] for zi in binary]
N = numpy.array(N)

binary_est = alpha + N.dot(gamma) > 0


# do it without classification trick
mma = svm_classify(img, y, 7.5)

