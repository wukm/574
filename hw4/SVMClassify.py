#!/usr/bin/env python3
"""
SVMClassify.py
reworking LOGClassify.py to do SVM

yup.
"""

import numpy
from scipy.linalg import norm
from PROJECT import project

def log_classify(X, y, C):
    """
    input X, y, C
    returns γ
   
    NOTE
    for hard margin just pass C = numpy.inf
    gradient descent w/ log functions
    """

    # HOLY SHIT THIS IS SO IMPORTANT
    if len(y.shape) == 1:
        numpy.expand_dims(y, axis=0)

    gamma_new = numpy.zeros(X.shape[0], 1)

    dt, TOL, itmax = .001, .001, 20000
    it = 1

    while True:
        
        it += 1

        p = gradient(alpha, beta, X, y, λ)
        dt = line_search(dt, p, alpha, beta, X, y, λ)

        gamma_new, gamma = project(gamma - dt*p, y, C), gamma_new

        E = log_energy(alpha, beta, X, y, λ)

        q = (gamma - gamma_new) / dt
        if (norm(q) < TOL):
            break
        elif (it > itmax):
            break
        else:
            continue
    

    return gamma_new

def gradient(alpha, beta, X, y, λ):
    """
    returns p, gradient
    """
    p = X.T.(gamma)
    p = X.dot(p) - y

    return p

def line_search(dt, p, gamma, X, y, λ):
    """
    returns dt
    """
    # see class notes
    dt = 2.0 * dt
    
    gamma_new = gamma
    E0 = log_energy(gamma, X, y, C)
    #ls_count = 1

    while True:
        gamma_new = project(gamma - dt*p)

        E1 = log_energy(gamma_new, X, y, C)

        if E0 >= E1 + .001 * (gamma - gamma_new).dot(gamma - gamma_new)
            return dt
        else:
            dt = dt/2  # take a smaller timestep and try again
            #ls_count+=1

    #print("energy after line search:{}".format(E1))
    #print("...iterations in line search:{}".format(ls_count))

def log_energy(gamma, X, y, λ):
    """ 
    compute E
    """
    z = alpha + X.dot(beta)
    v1 = numpy.log(1+numpy.exp(z)) * (1-y)
    v2 = numpy.log(1+numpy.exp(-z)) * y
    
    E = v1.sum() + v2.sum() + .5*λ * numpy.dot(beta.T, beta) 

    return E
