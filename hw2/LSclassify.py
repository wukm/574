#!/usr/bin/env python3
"""
LSClassify.py

Least squares clasification.
"""

import numpy
import scipy.linalg

def ls_classify(Xtrain, y, λ):
    """

    alpha, beta = ls_classify(Xtrain, y, λ)

    INPUT

    Xtrain -> training data 
    y   -> class labels for training data
    λ -> lambda value to use (`lambda` is a python builtin, haha)

    OUTPUT

    α -> scalar (offset of the separating hyperplane)
    β -> vector ('slope' of the separating hyperplane)
    
    a python implementation of LSclassify.m
    least squares classification given the training set Xtrain
    returns the parameters for the classifying hyperplane

    steps:

    (1) compute mean value centering
    (2) compute the SVD
    (3) compute alpha, beta

    """
    x_bar = Xtrain.mean(axis=0)  # average of each pixel (r,g,b); x_bar.size -> 3
    y_bar = y.mean() # average class label 
    
    # do mean centering (note: broadcasting)
    X_tilde = Xtrain - x_bar
    y_tilde = y - y_bar
    
    # full_matrices=False is the equivalent of MATLAB's "econ"
    # remember V is transposed of what it is in matlab
    U, S, V = scipy.linalg.svd(X_tilde, full_matrices=False)

    # c values
    C = S / (S*S + λ)

    # google "PEP 465" :)
    # check U.T and not U 
    β = numpy.dot(V.T, (C * numpy.dot(U.T, y_tilde)))
    α = y_bar - numpy.dot(x_bar, β)

    return α, β

if __name__ == "__main__":
    pass
