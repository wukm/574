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
    returns p, gradient
    """
    p = numpy.zeros(β.shape[0] +1, 1)

    # p[0] should be the derivative of E w.r.t. α
    # p[1:end] should be gradient of E w.r.t. β

    z = α + X.dot(β)

    # assuming it's faster to compute these once
    v1 = numpy.exp(z)
    v2 = numpy.exp(-z)
    
    # see class notes
    v1 = (v1 * (1 - y)) / (1 + v1)
    v2 = (v2 * y) / (1 + v2)

    v = v1 - v2

    p[0] = v.sum()
    p[1:] = numpy.dot(X.T, v) + λ*β
    
    return p

def line_search(dt, p, α, β, X, y, λ):
    """
    returns dt
    """
    # see class notes
    dt = 2.0 * dt

    E1 = log_energy(α, β, X, y, λ)
    
    while True:
        #update α, β via (2)
        α = α - (dt * p[0])
        β = β - (dt * p[1:])

        E1, E0 = log_energy(α, β, X, y, λ), E1

        if E1 > E0 - (.5*dt * numpy.dot(p.T, p)):
            break
        else:
            dt /= 2  # take a smaller timestep and try again

    return dt

def log_energy(α, β, X, y, λ):
    
    # compute E

    return E
