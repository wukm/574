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
    
    alpha = lasso(A, b, .001)

    return alpha

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

for i, row in enumerate(X):
    if not row.any():
        raise Exception("row {} is empty. abort.".format(i))
    row /= norm(row)
# D is now the transpose, so training points as columns
D = X.T
A = R.dot(D)
    


print('done loading finally')

# get alphas for each of the 32 images of individual 33
# via basis pursuit

λ=.001
class_ests = []
alphas = []
for row in Z:
    alpha = _get_alpha(row, R, A, λ)
    alphas.append(alpha)
    class_est = classify(D, alpha, row, lab) 
    class_ests.append(class_est)

correct = sum((x == 33 for x in class_ests))

from concentration import concentration

z2alphas = [_get_alpha(row, R, A, .001) for row in Z2]

conc = [concentration(alpha, lab) for alpha in z2alphas]

outliers_recognized = [c > .5 for c in conc]

print("the following outliers (individual 38) were (erronously) recognized")
print("(note: these are in order in the directory)")
print([i for i, c in enumerate(outliers_recognized) if c])
print("...or {} total.".format(sum(outliers_recognized)))

img33_recognized = [concentration(alpha, lab) > .5 for alpha in alpha]

print("the following images (individual 33) were (correctly) recognized")
print("(note: these are in order in the directory)")
print([i for i, c in enumerate(img33_recognized) if c])
print("...or {} total.".format(sum(img33conc)))
