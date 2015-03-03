#!/usr/bin/env python3

from scipy.io import loadmat
import os.path
from SVMClassify import svm_classify
import numpy

DATADIR = './data/SYNTHETIC/'
matfile = os.path.join((DATADIR, 'SYNTHETIC.txt')


d = loadmat(matfile)
X, y = d['X'], d['y']

C_vals = [.1, 1., 10., 100.]

for C in C_vals:
    gamma = svm_classify(X, y, C)
    beta = numpy.dot(X, gamma)
    
    # ugly
    foo = (0 < abs(gamma)) and (abs(gamma) < C)
    supp = gamma[foo]

    
