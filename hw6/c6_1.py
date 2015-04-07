#!/usr/bin/env python3

"""
HW6

facial recognition on the Extended YaleB dataset
using basis pursuit (lasso) with a dimensionality reduction step

note: i am still debugging LASSO.py. things are taking forever to load
for deterministic purposes, you can call seed(n) for some constant n
"""

from loadFACES import load_faces
from LASSO import lasso
from numpy.random import randn, seed
import numpy
from util import classify, normalize_columns
from scipy.linalg import norm

from sys import exit
# load faces 1-37 subsets as training data
X, lab = load_faces(list(range(1,38)), 'train')

# load face 33 subset as test
Z, lab_t = load_faces([33], 'test')

# load face 38 subset as an outlier (not in training set)
Z2, _ = load_faces([38], 'test')

# dimensionality reduction, 120 as new dimension is arbitrary
p = 120
R = randn(p,32256)
exit(0)
# normalize rows w.r.t. L2-norm
for row in R:
    row /= norm(row)

for row in X:
    row /= norm(row)
# D is now the transpose, so training points as columns
D = X.T
A = R.dot(D)
# get alphas for each of the 32 images of individual 33
# via basis pursuit

#λ=.001 # some more arbitrary parameter values
λ=.001

def _get_alpha(row, R, A, λ):
    # iterating over rows of an mxn matrix gives arrays with shape (n,)
    # and lasso expects (n,1). classic. see docstring within lasso.
    z = row.reshape((-1,1))

    # multiply by our random matrix for dim. reduction
    #A = R.dot(D) # this is just passed directly now
    b = R.dot(z)

    return lasso(A, b, λ, dt=.001, tol=.000001)

#alpha = _get_alpha(Z[0], R, R.dot(D), λ)
# yeah idk about putting such a computationally intensive function in a
# list comp
alphas = [_get_alpha(row, R, A, λ) for row in Z]
classests = [classify(D, alpha, z, lab) for alpha, z in zip(alphas, Z)]
# classify(D, alpha, z, lab) -> class_est
