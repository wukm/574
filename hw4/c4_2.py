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

d = loadmat(matfile)
img, lab = d['img'], d['lab']

#
#   make a gaussian M here
#
#   M(x,y) := exp{ - ||x-y||^2 / (2σ^2) }


# y contains class labels with y_i = {  1   x_i corresponds to a 9
#                                    { -1   else
y = (lab == 9).astype('double')
y = 2*y - 1 

gamma = kernel_classify(M, y, C=7.5)
