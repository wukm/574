#!/usr/bin/env python3
"""
SVMClassify.py
reworking LOGClassify.py to do SVM

probably bugs, but this is trying to be straightforward
"""

import numpy
from scipy.linalg import norm
from PROJECT import project

def kernel_classify(M, y, C, tol=.001):
    """
    input M, y, C
    returns γ
   
    NOTE
    for hard margin just pass C = numpy.inf
    gradient descent w/ log functions
    """

    # please get to the bottom of the dimensionality issues from before
    #if len(y.shape) == 1:
    #    numpy.expand_dims(y, axis=0)

    gamma = numpy.zeros((M.shape[0], 1))

    dt, itmax = .001, 20000
    it = 1

    while True:
        
        it += 1

        p = gradient(gamma, M, y)
        dt = line_search(dt, p, gamma, M, y, C)

        gamma_new = project(gamma - dt*p, y, C)

        q = (gamma_new - gamma) / dt

        if (norm(q) < tol):
            break
        elif (it > itmax):
            break
        else:
            gamma = gamma_new

    return gamma_new

def gradient(gamma, M, y):
    """
    grad(E(γ)) = X(X^T)γ - y
    """
    # do it in two steps, might save memory?

    return M.dot(gamma) - y

def line_search(dt, p, gamma, M, y, C):
    """
    returns dt
    """
    # see class notes
    dt = 2.0 * dt
    
    # initial energy and initial gamma do not change throughout
    E = _energy(gamma, M, y, C)

    while True:
        gamma_new = project(gamma - dt*p, y, C)

        E_new = _energy(gamma_new, M, y, C)

        if E >= E_new + .001 * (gamma - gamma_new).T.dot(gamma - gamma_new):
            return dt
        else:
            dt = dt/2  # take a smaller timestep and try again

def _energy(gamma, M, y, C):
    """ 
    E(γ) = ½ <γ, Mγ> - <γ, y>
    """
    first = M.dot(gamma)
    first = gamma.T.dot(first) / 2

    # fuck
    second = gamma.T.dot(y.T) 
    
    return first - second
