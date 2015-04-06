#!/usr/bin/env python3

"""
LASSO.py

a reworking of SVMClassify.py to do compressed sensing

i.e. to solve the minimization

min { E(x) | x in R^n }  with E(x) = λ||x||_1 + 1/2 (||Ax-b||_2)^2

also known as the 'lasso problem'

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
    a vector m (m.shape == z.shape), the solution to solution to
        min { dt*λ||x||_1 + 1/2 (||x-z||_2)^2 | x in R^n }

    the analytical solution is given by component-wise by
    
    m_i =  {    z_i - λ(dt)     if z_i > λ(dt)
           {    0               if -λ(dt) <= z_i <= λ(dt)
           {    z_i + λ(dt)     if z_i < -λ(dt)

    """

    c = λ * dt
    
    m =  (z - c)*((z - c) > 0) + (z + c)*((z + c) < 0)
    return m

def lasso_energy(x, A, b, λ):
    """
    outputs the scalar E(x), with 

    E(x) = λ||x||_1 + 1/2 (||Ax-b||_2)^2

    x in R^(n,1)
    A in R^(m,n)
    b in R^(m,1)

    output in R
    """
    #return λ*norm(x, ord=1) + .5 * (norm(A.dot(x) - b, ord=2)**2)
    return λ * sum(abs(x)) + .5 * (A.dot(x) - b).T.dot(A.dot(x) - b)
    
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

    return A.T.dot(A.dot(x) - b)

def line_search(dt, p, x, A, b, λ):
    """
    returns dt, a scalar

    please consider a max iteration, just for deterministic implementation

    i don't know what's going on here, i'm taking the iterated contiion
    wholesale.
    """
    # see class notes
    dt = 2.0 * dt
    
    # initial energy and initial x do not change
    E = lasso_energy(x, A, b, λ)

    while True:
        x_new = prox(x - dt*p, λ, dt)

        E_new = lasso_energy(x_new, A, b, λ)

        if not (E < E_new + .001*norm(x-x_new)**2):
            return dt
        else:
            dt = dt/2.  # take a smaller timestep and try again


def lasso(A, b, λ, dt=.001, tol=.000001):
    """
    solves the minimization

    min { E(x) | x in R^n }  with E(x) = λ||x||_1 + 1/2 (||Ax-b||_2)^2

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

    # does this help at all?
    A = A.astype('float64')
    b = b.astype('float64')

    # initial guess is zero in R^n
    x = numpy.zeros((A.shape[1], 1), dtype='float64')
    

    p = lasso_gradient(x, A, b)
    dt = line_search(dt, p, x, A, b, λ)

    for i in count():
        
        x_new = prox((x - dt*p), λ, dt)

        # normally it is divided by dt, but i do that below
        q = (x - x_new)

        if i > 0 and not i % 100:
            #print("i={}\tnorm(q)={}\tdt={}".format(i,norm(q),dt))


        # compensate for not dividing q by dt earlier
        if (norm(q) < tol*dt):
            break
        else:
            x = x_new
            p = lasso_gradient(x, A, b)
            dt = line_search(dt, p, x, A, b, λ)

        #print("iteration {}:".format(i))
        #print("\tdt={}".format(dt))
        #print("\tnorm(q)={}".format(norm(q)))

        #print("\tx={}".format(x.T))
        #print("\tp={}".format(p.T))

        #input()
        
        
    #print("done in {} iterations".format(i))
    #print("norm(q)={} < tol={}".format(norm(q), tol))

    return x_new
