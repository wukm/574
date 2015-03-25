#!/usr/bin/env python3

import numpy


def as_column_vector(v):
    """
    v is any iterable. returns an ndarray with shape ( len(v), 1)
    """
    return numpy.array(v, ndmin=2).T

def row_norm_matrix(A, shape=None):
    """
    needed for forming M
    if A is an nxd matrix, return an MxN matrix that
    looks like

    [[ ||x1||^2 ||x1||^2 ... ||x1||^2 ]
     [ ||x2||^2 ||x2||^2 ... ||x2||^2 ]
     ...
     [ ||xM||^2 ||xM||^2 ... ||xM||^2 ]]

    where xi represents a row of X.

    this subfunction is useful for making the gaussian kernel and
    testing whats-it matrix N
    """
   
    # really making shape a tuple is just to make the interface more foolproof
    assert shape[0] == A.shape[0], "can't broadcast to this shape!"

    n = shape[1]

    # a flattened list of row two norms of rows in A (data points)
    row_norms = [row.T.dot(row) for row in A]

    Q = as_column_vector(row_norms)
    Q = numpy.tile(Q, (1, shape[1]))

    return Q

def gaussian_M(X, sigma):
    """
    returns the matrix M with M_(i,j) = M(x_i, x_j) 

    where M(x,y) := exp{ - ||x-y||^2 / (2σ^2) }
    uses some linear algebra tricks.

    """
    n = X.shape[0] 
    Q = row_norm_matrix(X, (n,n))

    xxt = X.dot(X.T)

    M = Q - xxt

    M = M + M.T  # now M contains ||x_i - x_j||^2

    M = -M / (2*sigma**2)

    return numpy.exp(M)

def gaussian_N(Z, X, sigma):
    
    m, d1 = Z.shape
    n, d2 = X.shape

    assert d1 == d2

    QZ = row_norm_matrix(Z, (m,n))
    QX = row_norm_matrix(X, (n,m))

    N = QZ - 2*Z.dot(X.T) + QX.T

    N = -N / (2*sigma**2)

    return numpy.exp(N)

if __name__ == "__main__":

    from loadISOLET import load_isolet

    vowels = [1, 5, 9, 15, 21]
    sigma = 9
    voice, lab = load_isolet(vowels, 'train')
    voice_t, lab_t = load_isolet(vowels, 'test')

    M = gaussian_M(voice, sigma)
    N = gaussian_N(voice_t, voice, sigma)

    # now do multiclass SVM
    # ...
