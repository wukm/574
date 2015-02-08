#!/usr/bin/env python3

import sys
import os
import scipy.sparse.linalg
import numpy
import PIL.Image
from loadMNIST import *
from opentiff import normalize_grayscale

def rank_two_approx(A, return_parts=False):
    """
    input:  A <- a matrix,
            return_parts <- bool        
    output: approximated matrix 
            parts (optional)

    returns the approximated matrix, given by the rank two approximation of A.
    if return_parts is specified, the tuple (u,s,v) is returned as well.

    NOTE:   (Look into this) a minor bother is that the *second* singular value is
    actually greater than the first -- it seems that scipy.sparse.svds outputs
    in opposite order (compared to our lecture notes)
            Also, V is non transposed. 
    """
    u, s, v = scipy.sparse.linalg.svds(A, k=2, which='LM')
    
    # A ≈ [U][Σ][V^T]
    approx = u.dot(s*numpy.eye(2)).dot(v)
    
    if return_parts:
        return approx, (u, s, v)
    else:
        return approx

if __name__ == "__main__":

    assert os.path.isdir(DATA_DIR), "cannot find data directory in {}".format(DATA_DIR)
        
    for d in (1,2):
        A = load_all(d)
        AA, (u, s, v) = rank_two_approx(A, return_parts=True)
       
        # convert right singular vectors to image shape
        v1 = v[0].reshape((28,28))
        v2 = v[1].reshape((28,28))

        # because of dumbness in the interface, v2 is actually the right
        # singular vector corresponding to the largest singular value
        v2, v1 = v1, v2

        v1 = normalize_grayscale(v1)
        v2 = normalize_grayscale(v2)
        
        # convert to uint8 again for easier writing
        v1 = v1.astype('uint8')
        v2 = v2.astype('uint8')
        
        file_base = "all_{}_{}.tiff"
        Image.fromarray(v1).show()
        #Image.fromarray(v1).save(file_base.format(d, 'v1'))
        #Image.fromarray(v2).save(file_base.format(d, 'v2'))
        
        # see comment above
        u1 = u.T[1]
        u2 = u.T[0]

        # returns index of maximum / minimum in question
        ibig, ismall = u2.argmax(), u1.argmin()
        
        big = A[ibig].reshape((28,28))
        small = A[ismall].reshape((28,28))

        big = normalize_grayscale(big)
        small = normalize_grayscale(small)

        big = big.astype('uint8')
        small = small.astype('uint8')

        #Image.fromarray(big).save(file_base.format(d, 'big'))
        #Image.fromarray(small).save(file_base.format(d, 'small'))



