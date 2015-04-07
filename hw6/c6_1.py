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
from numpy.random import randn, seed, get_state
import numpy
from util import classify
from scipy.linalg import norm

from sys import exit

def nonzeros(a):
    """
    this is just a convenience function. for printing out nonzero entries of an
    array. the reshaping is for visual inspection
    """
    return a[a.nonzero()].reshape((1,-1))

def _get_alpha(row, R, A, λ):
    # iterating over rows of an mxn matrix gives arrays with shape (n,)
    # and lasso expects (n,1). classic. see docstring within lasso.
    z = row.reshape((-1,1))

    # multiply by our random matrix for dim. reduction
    #A = R.dot(D) # this is just passed directly now
    b = R.dot(z)

    return lasso(A, b, λ, dt=.001, tol=.000001)

def problem_setup(rand_seed=None):
    # load faces 1-37 subsets as training data
    X, lab = load_faces(list(range(1,38)), 'train')

    # load face 33 subset as test
    Z, lab_t = load_faces([33], 'test')

    # load face 38 subset as an outlier (not in training set)
    Z2, _ = load_faces([38], 'test')

    # dimensionality reduction, 120 as new dimension is arbitrary
    #if rand_seed is not None:
    #    seed(rand_seed)

    p = 120
    R = randn(p,32256)
    
    # normalize rows w.r.t. L2-norm
    for row in R:
        row /= norm(row)

    for row in X:
        row /= norm(row)
    # D is now the transpose, so training points as columns
    D = X.T
    A = R.dot(D)
    
    state = get_state() 
    return X, lab, Z, lab_t, Z2, R, D, A

X, lab, Z, lab_t, Z2, R, D, A = problem_setup()

# get alphas for each of the 32 images of individual 33
# via basis pursuit

λ=.001
class_ests = []
for row in Z:
    alpha = _get_alpha(Z[0], R, A, λ)
    class_est = classify(D, alpha, row, lab) 
    class_ests.append(class_est)


# if you want list comps instead
#alphas = [_get_alpha(row, R, A, λ) for row in Z]
#classests = [classify(D, alpha, z, lab) for alpha, z in zip(alphas, Z)]
