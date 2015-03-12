#!/usr//bin/env python3

"""
PROJECT.py

porting of MATLAB function PROJECT.m

"""
import numpy as np

def project(x,y,C):
    """
    compute the projection of the n-vector x onto the convex
    set K defined via
        C >= x_i y_i >= 0,
        sum(x) = 0
    using Dykstra's alternating projection algorithm
    """

    p = np.zeros((x.shape[0], 1))
    q = np.zeros((x.shape[0], 1))

    tol = 1e-15 
    while True:
        p_old, q_old = p, q

        # project and calculate increment
        z = _proj_c(x + p)
        p = x + p - z

        # project and calculate increment
        x = _proj_d((z+q), y, C)
        q = z + q - x

        nrm = (p - p_old).T.dot(p - p_old) + (q - q_old).T.dot(q - q_old)

        if nrm < tol:
            break

    return x

def _proj_c(x):
    return x - x.mean()

def _proj_d(x, y, C):
    
    if np.isinf(C):
        m = x * (x * y >= 0)
    elif C > 0:
        cond1 = x*y >= 0
        cond2 = x*y <= C
        m = x * cond1 * cond2 + C * y * (1 - cond2)
    else:
        raise Exception("invalid penalty parameter C: {}".format(C))

    return m
