#!/usr/bin/env python3

"""
using SVM to find a separating hyperplane
"""
from scipy.io import loadmat
import os.path
from SVMClassify import svm_classify
from util import gamma_to_hyperplane
import numpy

# question 3b -- loading synthetic data 
DATADIR = './data/SYNTHETIC/'
matfile = os.path.join(DATADIR, 'SYNTHETIC.mat')

# get X and y from file
d = loadmat(matfile)
X, y = d['X'], d['y']

to_write = []

for C in [.1, 1., 10., 100., numpy.inf]:

    gamma = svm_classify(X, y, C)
    
    alpha, beta = gamma_to_hyperplane(gamma, X, y, C)

    to_write.append("C={}".format(C))
    to_write.append("α={}".format(alpha))
    # print beta too if it's not too unwieldy
    if beta.size < 5:
        to_write.append("β={}".format(beta.flatten()))
   
    z = y * (alpha + X.dot(beta)) 

    valid = (z >= 1).sum()
    frac_valid = valid / z.size

    to_write.append("satisfied {} out of {} (= {}%) constraints".format(valid,
        z.size, frac_valid))

    to_write.append('\n')
    
lines = '\n'.join(to_write)

with open('c4_1.log', 'w') as f:
    f.write(lines)
