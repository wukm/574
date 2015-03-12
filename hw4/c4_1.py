#!/usr/bin/env python3

from scipy.io import loadmat
import os.path
from SVMClassify import svm_classify
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
    beta = numpy.dot(X.T, gamma)

    # get a boolean array, same shape as gamma
    # true when 0 < |γ_i| < C
    supp = (0 < abs(gamma)) & (abs(gamma) < C)

    # i still don't fucking get row filtering in numpy
    # but this actually gets right dimensions. what??
    s = numpy.nonzero(supp)[0]

    alpha = (y[s] - X[s].dot(beta)).sum() / s.size

    to_write.append("C={}".format(C))
    to_write.append("α={}".format(alpha))
    to_write.append("β={}".format( beta.flatten()))
   
    z = y * (alpha + X.dot(beta)) 

    valid = (z >= 1).sum()
    frac_valid = valid / z.size

    to_write.append("satisfied {} out of {} (= {}%) constraints".format(valid,
        z.size, frac_valid))

    to_write.append('\n')
    
lines = '\n'.join(to_write)

with open('c4_1.txt', 'w') as f:
    f.write(lines)
