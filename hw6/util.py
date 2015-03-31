#!/usr/bin/env python3

from scipy.linalg import norm
import numpy

def normalize_columns(D):
    """
    each column will have L2 norm == 1
    """

    for col in D.T:
        col /= norm(col)

    return D

def classify(D, alpha, z, lab):

    classes = set(lab)
    f = []  
    # get relevant things
    for label in classes:
        v = numpy.nonzero(lab == label)[0]
        z_est = D[:,v].dot(alpha[v])
        f.append((label, norm(z-zest)))
    
    class_est = min(f, key=lambda x: x[1])
    
    #print("class found", class_est[0], "with error", class_est[1])
    return class_est[0]
