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
from SVMClassify import svm_classify
import numpy


# when refactoring, just put this in a function that returns X,y 
DATADIR = './data/'
matfile = os.path.join(DATADIR, 'MNIST_SMALL.mat')

def as_column_vector(v):
    """
    v is any iterable. returns an ndarray with shape ( len(v), 1)
    """
    return numpy.array(v, ndmin=2).T

def _make_row_norm_matrix(X):
    """
    needed for forming M
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

    # a flattened list of row two norms of rows in X (data points)
    x_norms = [row.T.dot(row) for row in X]

    Q = as_column_vector(x_norms)
    Q = numpy.tile(Q, (1, X.shape[0]))

    return Q

def _make_gaussian_kernel(X, sigma):
    """
    returns the matrix M with M_(i,j) = M(x_i, x_j) 

    where M(x,y) := exp{ - ||x-y||^2 / (2σ^2) }
    uses some linear algebra tricks.

    """
    
    Q = _make_row_norm_matrix(X)

    xxt = X.dot(X.T)

    M = Q - xxt

    M = M + M.T  # now M contains ||x_i - x_j||^2

    M = -M / (2*sigma**2)

    return numpy.exp(M)

if __name__ == "__main__":

    sigma = 5

    d = loadmat(matfile)
    img, lab = d['img'], d['lab']

    # this loop takes 5 seconds
    try:
        M = numpy.load('c4_2_gaussM.pickle')
    except FileNotFoundError:
        M = _make_gaussian_kernel(img, sigma)
        M.dump('c4_2_gaussM.pickle')

    # y contains class labels with y_i = {  1   x_i corresponds to a 9
    #                                    { -1   else
    y = (lab == 9).astype('double')
    y = 2*y - 1     # normalize to class labels {-1, 1}

    C = 7.5
    gamma = kernel_classify(M, y, C)

    # classifying step
    supp = (0 < abs(gamma)) & (abs(gamma) < C)

    s = numpy.nonzero(supp)[0]
    v = M.dot(gamma)
    alpha = (y[s] - v[s]).sum() / s.size

    # training portion is done now --- we have alpha and beta.

    img_t, lab_t = load_mnist([4,7,9], 'test')
    lab_t = numpy.array(lab_t)
    binary = numpy.array((lab_t == 9), ndmin=2).T  # again, make it a column vector


    sigma = 5

    # FORMING N THIS WAY TAKES ABOUT 45 SECONDS
    try:
        N = numpy.load('c4_2_gaussN.pickle')
    except FileNotFoundError:
        N =  [[ (zi - xj).T.dot(zi - xj) / (2 * sigma**2) for xj in img] for zi in binary]
        N = numpy.array(N)
        N.dump('c4_2_gaussN.pickle')

    binary_est = (alpha + N.dot(gamma)) > 0

    accuracy = binary_est.sum() / binary_est.size


    # DEBUG
    print("alpha", alpha)
    print("accuracy", accuracy)

    gamma2 = svm_classify(img, y, C=7.5, tol=.03, dt=.001)
    beta = img.T.dot(gamma2)
    supp2 = (0 < abs(gamma2)) & (abs(gamma2) < C)
    s2 = numpy.nonzero(supp)[0]

