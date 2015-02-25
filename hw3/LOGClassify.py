#!/usr/bin/env python3
"""
LOGClassify.py
an implementation/port of LOGClassify.m
"""

import numpy
from scipy.linalg import norm

def log_classify(X, y, λ):
    """
    input X, y, λ
    returns α, β
    
    gradient descent w/ log functions
    """

    # HOLY SHIT THIS IS SO IMPORTANT
    if len(y.shape) == 1:
        numpy.expand_dims(y, axis=0)

    alpha, beta = 0, numpy.zeros(X.shape[1])

    flag, dt, it, TOL, itmax = False, .001, 1, .001, 20000

    while not flag:
        
        it += 1

        print("starting iteration {}".format(it))
        p = gradient(alpha, beta, X, y, λ)
        print('norm(p)={}'.format(norm(p)))
        print(p)
        dt = line_search(dt, p, alpha, beta, X, y, λ)
        print('dt={}'.format(dt))

        alpha = alpha - (dt * p[0])
        beta = beta - (dt * p[1:])
        E = log_energy(alpha, beta, X, y, λ)
        print("energy is currently: {}".format(E))

        if (norm(p) < TOL):
            print("tolerance reached")
            break
        elif (it > itmax):
            print("max iterations reached")
            break
        else:
            continue
    

    return alpha, beta

def _get_v(alpha, beta, X, y):
    """
    NOTE THIS ONLY WORKS FOR THE GRADIENT
    THE ENERGY v1, v2 ARE LITERALLY DIFFERENT.
    returns v1, v2. see class notes.
    """

    z = alpha + X.dot(beta)
    #print("got z")
    # assuming it's faster to compute these once
    z1 = numpy.exp(z)
    z2 = numpy.exp(-z)
    #print("got z1,z2")
    # see class notes
    v1 = (z1 * (1 - y)) / (1 + z1) 
    v2 = (z2 * y) / (1 + z2)
    return v1, v2

def gradient(alpha, beta, X, y, λ):
    """
    returns p, gradient
    """
    p = numpy.zeros((beta.shape[0] +1))

    # p[0] should be the derivative of E w.r.t. α
    # p[1:end] should be gradient of E w.r.t. β
    #print('need to get v')
    v1, v2 = _get_v(alpha, beta, X, y) 
    #print('got v1,v2')
    v = v1 - v2
    p[0] = v.sum()
    p[1:] = numpy.dot(X.T, v) + λ*beta
    
    return p

def line_search(dt, p, alpha, beta, X, y, λ):
    """
    returns dt
    """
    # see class notes
    dt = 2.0 * dt
    
    alpha_new, beta_new = alpha, beta
    E0 = log_energy(alpha, beta, X, y, λ)
    ls_count = 1

    while True:
        #update α, β via (2)
        alpha_new = alpha - (dt * p[0])
        beta_new = beta - (dt * p[1:])

        #E1, E0 = log_energy(alpha, beta, X, y, λ), E1
        E1 = log_energy(alpha_new, beta_new, X, y, λ)

        if E1 <= E0 - (.5*dt * numpy.dot(p.T, p)):
            return dt
        else:
            dt = dt/2  # take a smaller timestep and try again
            ls_count+=1

    print("energy after line search:{}".format(E1))
    print("...iterations in line search:{}".format(ls_count))

def log_energy(alpha, beta, X, y, λ):
    """ 
    compute E
    HOLY SHIT, NEED DIFFERENT V'S
    v1, v2 = _get_v(alpha, beta, X, y)  BAD BAD BAD
    """
    z = alpha + X.dot(beta)
    v1 = numpy.log(1+numpy.exp(z)) * (1-y)
    v2 = numpy.log(1+numpy.exp(-z)) * y
    
    E = v1.sum() + v2.sum() + .5*λ * numpy.dot(beta.T, beta) 

    return E
