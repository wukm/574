#!/usr/bin/env python3

"""
c4_2.py

This solves HW #4 coding II
"""

from scipy.io import loadmat
from loadMNIST import load as load_mnist
import os.path

from PROJECT import project 
from kernelSVM import kernel_classify
from SVMClassify import svm_classify
from kernel import gaussian_M, gaussian_N
from util import gamma_to_hyperplane

import numpy


# when refactoring, just put this in a function that returns X,y 
DATADIR = './data/'
matfile = os.path.join(DATADIR, 'MNIST_SMALL.mat')

def as_column_vector(v):
    """
    v is any iterable. returns an ndarray with shape ( len(v), 1)
    """
    return numpy.array(v, ndmin=2).T


if __name__ == "__main__":

    sigma = 5

    d = loadmat(matfile)
    img, lab = d['img'], d['lab']

    # this loop takes 5 seconds
    try:
        M = numpy.load('c4_2_gaussM.pickle')
    except FileNotFoundError:
        M = gaussian_M(img, sigma)
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

    # this is the 'cheating' result to compare to
    binary = as_column_vector(lab_t == 9)

    sigma = 5

    # FORMING N THIS WAY TAKES ABOUT 45 SECONDS
    try:
        N = numpy.load('c4_2_gaussN.pickle')
    except FileNotFoundError:
        #N =  [[ (zi - xj).T.dot(zi - xj) / (2 * sigma**2) for xj in img] for zi in binary]
        #N = numpy.array(N)
        N = gaussian_N(img_t, img, sigma)
        N.dump('c4_2_gaussN.pickle')

    binary_est = (alpha + N.dot(gamma)) > 0
    
    assert binary.shape == binary_est.shape

    accuracy = sum(binary_est == binary) / binary_est.size


    # DEBUG
    print("kernel SVM accuracy:\t", accuracy[0])

    # then part f, this should really be separate
    
    C = 7.5
    try:
        gamma2 = numpy.load('c4_2_gamma2.pickle')
    except FileNotFoundError:
        gamma2 = svm_classify(img, y, C, tol=.03, dt=.001)
        gamma2.dump('c4_2_gamma2.pickle')
    
    alpha, beta = gamma_to_hyperplane(gamma, img, y, C)

    binary_est2 = (alpha + img_t.dot(beta)) > 0
    
    assert binary.shape == binary_est2.shape

    accuracy2 = sum(binary_est2 == binary) / binary_est2.size

    print("(non-kernel) SVM accuracy:\t", accuracy2[0])
