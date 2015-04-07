#!/usr/bin/env python3

"""
make this a unit test.

this should result in an x_est s.t.

nonzeros(x_est) = [a] where a is very close to 1 (i.e. .99)
and other items are 0 or very close
"""

import numpy
from numpy.random import randn
from scipy.linalg import norm
from LASSO import *

def nonzeros(a):

    return a[a.nonzero()].reshape((1,-1))

x = numpy.zeros((500,1))
x[2] = [1.]
A = randn(50,500)

# normalize by row
for row in A:
    row /= norm(row)

b = A.dot(x)

x_est = lasso(A,b,.001)
