#!/usr/bin/env python3

"""
LASSO.py

a reworking of SVMClassify.py to do compression sensing minimization
yadda yadda
"""

import numpy
from itertools import count

def prox(z, λ, dt):
    """
    λ parameter
    dt step
    z a vector (row or column)

    outputs a vector m using fancy tricks
    
    ***add words here***

    """

    c = λ * dt
    
    return (z - c) * (z - c > 0) + (z + c) * (z + c < 0)


def lasso_energy(x, A, b, λ):
    """
    E(x) = λ||x||_1 + 1/2 (||Ax-b||_2)^2
    """
    # alernatively:
    # from scipy.linalg import norm
    #return λ + norm(x, ord=1) + .5 * norm(A.dot(x) - b, ord=2)

    return λ * sum(abs(x)) + .5 * (A.dot(x) - b).dot(A.dot(x) - b)
    
def lasso_gradient(x, A, b):
    """
    g(x) = 1/2 ||Ax -b||^2

    => grad(g(x)) = A^T (Ax - b)
    
    just to L2 norm part, the L1 norm isn't differentiable
    """

    return A.T.dot(A.dot(x) - b)

def line_search(dt, p, x, A, b, λ):
    """
    returns dt
    """
    # see class notes
    dt = 2.0 * dt
    
    # initial energy and initial gamma do not change throughout
    E = lasso_energy(x, A, b, λ)

    while True:
        x_new = prox(x - dt*p, λ, dt)

        E_new = lasso_energy(x_new, X, b, λ)

        if E >= E_new + .001 * (x- x_new).T.dot(x - x_new):
            return dt
        else:
            dt = dt/2  # take a smaller timestep and try again

def lasso(A, b, λ, dt=.001, tol=.000001):
    """
    solves the compressed sensing minimization dealie-o
    add words here
    """

    x = numpy.zeros((A.shape[1], 1))
    
    # check out my fancy trick! 
    for i in count():
        
        p = lasso_gradient(x, A, b)
        dt = line_search(dt, p, x, A, b, λ)

        x_new = prox((x - dt*p), λ, dt)

        q = (x_new - x) / dt

        if (norm(q) < TOL):
            break
        else:
            x = x_new
    
    print("done in {} iterations".format(i))

    return x_new
