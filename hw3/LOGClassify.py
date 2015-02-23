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

    α, β = 0, numpy.zeros(X.shape[1], 1)

    flag, dt, iter, TOL, itmax = False, .001, 1, .001, 20000

    while not flag:
        
        iter += 1

        p = gradient(α, β, X, y, λ)
        dt = line_search(dt, p, α, β, X, y, λ)

        # update alpha, beta
        # α, β = 

        if (norm(p) < TOL):
            print("tolerance reached")
            flag = True
        elif (iter > itmax):
            print("too many iterations, abort")
            flag = True
        else:
            pass # continue

    return α, β

def gradient(α, β, X, y, λ):
    """
    returns p
    """
    p = numpy.zeros(β.shape[0] +1, 1)

    # p[0] should be the derivative of E w.r.t. α
    # p[1:end] should be gradient of E w.r.t. β

    # p[0] = 
    # p[1:] = 

    return p

def line_search(dt, p, α, β, X, y, λ):
    """
    returns dt
    """
    dt = 2.0 * dt

    E = log_energy(α, β, X, y, λ)
    
    #update α, β via (2)
    α = α - (dt * p[0])
    β = β - (dt * p[1:])

    E_new = log_energy(α, β, X, y, λ)

    while E_new > E - .5*dt * numpy.dot(p.T, p):

        # take a smaller timestep and try again
        #
        #
        #

        E_new = log_energy(α, β, X, y, λ)

    return dt

def log_energy(α, β, X, y, λ):
    
    # compute E

    return E
