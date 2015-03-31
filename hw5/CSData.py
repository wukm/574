#!/usr/bin/env python3

from scipy.fftpack import dct
from random import randint, sample
from numpy import eye, zeros
from numpy.random import randn

def cs_data(m, n, s):
    """
    BE SUPER FUCKING CAREFUL THAT YOU ARE PASSING THE RIGHT VALUES HERE.
    it seems that matlab is inclusive ranges and python is not so i can just
    use the default values and be okay. pay attention!
    """
    

    A = dct(eye(n))
    
    # wheeeee
    rows = sample(range(n), m)
    
    A = A[rows]

    x_ex = zeros((n,1))
    inds = sample(range(n), s)
    x_ex[inds] = randn(s,1)

    b = A.dot(x_ex)
    return A, b, x_ex
    
    
    

