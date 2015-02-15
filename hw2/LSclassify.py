#!/usr/bin/env python3
"""
LSClassify.py

Least squares clasification.
"""


def ls_classify(Xtrain, y, lamb):
    """
    alpha, beta = ls_classify(Xtrain, y, lamb)

    INPUT

    Xtrain -> training data 
    y   -> class labels for training data
    lamb -> lambda value to use

    OUTPUT

    alpha   -> scalar
    beta    -> vector
    
    a python implementation of LSclassify.m
    least squares classification given the training set Xtrain
    returns the parameters for the classifying hyperplane

    steps:

    (1) compute mean value centering
    (2) compute the SVD
    (3) compute alpha, beta

    real comments here later.
    """
    pass
    #return alpha, beta
