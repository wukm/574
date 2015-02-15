#!/usr/bin/env python3
"""
LSClassify.py

Least squares clasification.
"""

import numpy
#import scipy.sparse.linalg as sla
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

    real comments here later.
    """
    # return α, β
    
    x_bar = Xtrain.mean(axis=0) # average for pixels; x_bar.size -> 3
    y_bar = y.mean() # average class label 
    
    #subtract x_bar from each row in Xtrain. broadcasting!
    X_tilde = Xtrain - x_bar
    # ditto
    y_tilde = y - y_bar
    
    # woo, I don't have to use sparse!
    # full_matrices=False is equivalent of the matlab 'econ'
    U, S, V = scipy.linalg.svd(X_tilde, full_matrices=False)

    # β = V(Σ² + λI)⁻¹Σ(U.T)(y_tilde)
    c_values = S / (S*S + λ)

    # can't wait for the @ operator in python 3.5
    β = numpy.dot(V.T, (c_values * numpy.dot(U.T, y_tilde)))
    α = y_bar - numpy.dot(x_bar, β)

    return α, β

if __name__ == "__main__":
    pass
