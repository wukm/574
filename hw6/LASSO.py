#!/usr/bin/env python3

"""
LASSO.py

a reworking of SVMClassify.py to do compressed sensing

i.e. to solve the minimization
min { E(x) | x in R^n } with E(x) = λf(x) + g(x)
        f(x) = ||x||_1
        g(x) = 1/2 (||Ax-b||_2)^2

this is also known as the 'lasso problem' or 'basis pursuit'

There is only one real public function here. import it (for now) by

from LASSO import lasso
"""

import numpy
from itertools import count
from scipy.linalg import norm

def prox(z, λ, dt):
    """
    λ:  parameter
    dt: step
    z:  a vector (row or column)
    
    output:
    a vector m (m.shape == z.shape), 
        m = arg min { dt*λ||x||_1 + 1/2 (||x-z||_2)^2 | x in R^n }

    the analytical solution is given by component-wise by
    
    m_i =  {    z_i - λ(dt)     if z_i > λ(dt)
           {    0               if -λ(dt) <= z_i <= λ(dt)
           {    z_i + λ(dt)     if z_i < -λ(dt)

    """

    c = λ * dt
    
    # this is equivalent to the analytical sol'n above (check it)
    m = (z-c > 0.0).astype('double')*(z-c) + (z+c < 0.0).astype('double')*(z+c)

    return m

def lasso_energy(x, A, b, λ):
    """
    INPUT 

    the arrays x, A, b
    x in R^(n×1) (e.g. is usually the iterated approximation)
    A in R^(m×n)
    b in R^(m×1)

    OUTPUT

    the scalar E(x) = λ||x||_1 + 1/2 (||Ax-b||_2)^2
    """
    # these are equivalent, use whichever
    energy = λ*norm(x, ord=1) + .5 * (norm(A.dot(x) - b, ord=2)**2)
    #energy = λ * sum(abs(x)) + .5 * (A.dot(x) - b).T.dot(A.dot(x) - b)
   
    return energy

def lasso_gradient(x, A, b):
    """
    g(x) = 1/2 ||Ax -b||^2

    => grad(g(x)) = A^T (Ax - b)
    
    which is just L2 norm part, as the L1 norm isn't differentiable

    x in R^(n,1)
    A in R^(m,n)
    b in R^(m,1)

    output in R^(n,1)
    """
    p = A.dot(x) - b
    p = A.T.dot(p)
    return p

def line_search(dt, p, x, A, b, λ):
    """
    returns dt, a scalar

    please consider a max iteration, just for deterministic implementation

    i don't know what's going on here tbh. this is related to a general
    backtracking line search algorithm (with Armijo-Goldstein condition)

    en.wikipedia.org/wiki/Backtracking_line_search
    """
    # rather than starting in the same place, this will actually increase the
    # last used step size in the new direction (-p)
    # while we can get away with it
    dt *= 2.0
    
    # initial energy and initial x and initial p do not change
    E = lasso_energy(x, A, b, λ)
    
    c = .001 # this is an arbitrary parameter in (0,1)

    while True:
        x_new = prox(x - dt*p, λ, dt)

        E_new = lasso_energy(x_new, A, b, λ)

        if not (E < E_new + c*norm(x-x_new)**2):
            return dt
        else:
            dt /= 2.  # take a smaller timestep and try again


def lasso(A, b, λ, dt=.001, tol=.000001):
    """
    returns x = arg min { E(x) | x in R^n }
        with E(x) = λ||x||_1 + 1/2 (||Ax-b||_2)^2

    also known as the 'lasso problem' using gradient descent with the initial
    approximation 0 in R^(nx1).

    INPUT:

    A: an mxn numpy.ndarray
    b: an mx1 numpy.ndarray (will raise AssertionError if wrong shape!)

    λ: a parameter (real number)
    dt: initial timestep (default .001)
    tol: stopping tolerance (default .000001)
    
    OUTPUT:

    x: an nx1 numpy.ndarray

    """
    
    # maybe I should just fix the shape of b if it's wrong, because it's often
    # annoying to have to do it in the main loop. "right-ish" shape could either
    # be (m,) or (1,m), or b could not be an array yet. this would fix the
    # problem regardless, i believe.
    # b = numpy.array(b).reshape(-1,1)

    # initial guess is zero in R^n
    x = numpy.zeros((A.shape[1], 1), dtype='float64')
    


    # if you want a MAX_ITERATIONS, use
    # for i in range(MAX_ITERATIONS):
    for i in count():
        
        p = lasso_gradient(x, A, b)
        dt = line_search(dt, p, x, A, b, λ)

        x_new = prox(x - dt*p, λ, dt)


        #if not i % 1000 and i > 0:
        #    print("i={}\tnorm(q)={}\tdt={}".format(i,norm(q),dt))
        #    #pass


        # to be used in the stopping condition ||q|| < ε
        # an alternative stopping condition would be || Ax - b || < ε
        q = (x_new - x) / dt

        if (norm(q) < tol):
            break
        else:
            x = x_new

        #print("iteration {}:".format(i))
        #print("\tdt={}".format(dt))
        #print("\tnorm(q)={}".format(norm(q)))

        #print("\tx={}".format(x.T))
        #print("\tp={}".format(p.T))

        #input()
        
        
    print("done in {} iterations".format(i))
    #print("norm(q)={} < tol={}".format(norm(q), tol))

    return x_new # return the latest iteration you have
