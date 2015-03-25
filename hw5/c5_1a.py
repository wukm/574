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
import numpy

from kernel import gaussian_N, gaussian_M, as_column_vector

# when refactoring, just put this in a function that returns X,y 
DATADIR = './data/'
matfile = os.path.join(DATADIR, 'MNIST_SMALL.mat')


if __name__ == "__main__":

    sigma = 5
    C = 7.5

    d = loadmat(matfile)
    img, lab = d['img'], d['lab']

    try:
        M = numpy.load('c5_1a_gaussM.pickle')
    except FileNotFoundError:
        M = gaussian_M(img, sigma)
        M.dump('c5_1a_gaussM.pickle')

    # y contains class labels with y_i = {  1   x_i corresponds to a 9
    #                                    { -1   else
    y = (lab == 9).astype('double')
    y = 2*y - 1     # normalize to class labels {-1, 1}

    gamma = kernel_classify(M, y, C)

    # classifying step
    supp = (0 < abs(gamma)) & (abs(gamma) < C)

    s = numpy.nonzero(supp)[0]
    v = M.dot(gamma)
    alpha = (y[s] - v[s]).sum() / s.size

    # training portion is done now --- we have alpha and beta.

    img_t, lab_t = load_mnist([4,7,9], 'test')
    lab_t = numpy.array(lab_t)
    binary = as_column_vector(lab_t == 9)

    try:
        N = numpy.load('c5_1a_gaussN.pickle')
    except FileNotFoundError:
        N = gaussian_N(img_t, img, sigma)
        N.dump('c5_1a_gaussN.pickle')

    binary_est = (alpha + N.dot(gamma)) > 0

    accuracy = binary_est.sum() / binary_est.size


    # DEBUG
    print("alpha", alpha)
    print("accuracy", accuracy)
