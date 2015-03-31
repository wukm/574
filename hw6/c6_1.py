#!/usr/bin/env python3

"""
HW6

using a lasso technique to do facial recognition on the Extended YaleB dataset
"""

from loadFACES import load_faces
from LASSO import lasso
from numpy.random import randn
import numpy

X, lab = load_faces(list(range(1,38)), 'train')
Z, lab_t = load_faces([33], 'test')

Z2, _ = load_faces([38], 'test') #outlier

# arbitrarily decide a reduced dimension for the data points
p = 120
R = randn(p,32256)

# normalize rows w.r.t. l2-norm
for row in R:
    row /= sum(row**2)

# get alphas for each of the 32 images of individual 33
λ=.001 # some more arbitrary parameter values
D = X.T # pretty sure this is all D is

for row in Z:
    z = row.reshape((-1,1)) # gaah

    A = R.dot(D)
    b = R.dot(z)
    alpha = lasso(A, b, λ)
    break

