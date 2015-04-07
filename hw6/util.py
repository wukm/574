#!/usr/bin/env python3

from scipy.linalg import norm
import numpy

def classify(D, alpha, z, lab):

    # now you can pass rows directly from a matrix
    if z.ndim == 1:
        z = z.reshape((-1,1))

    classes = numpy.unique(lab)
    f = []  
    # get relevant things
    for label in classes:
        v = numpy.nonzero(lab == label)[0]
        z_est = D[:,v].dot(alpha[v])
        f.append((label, norm(z-z_est)))
    
    class_est = min(f, key=lambda x: x[1])
    
    print("class found", class_est[0], "with error", class_est[1])
    return class_est[0]
