#!/usr/bin/env python3

import numpy
from scipy.linalg import norm

def concentration(alpha, lab):
    
    classes = numpy.unique(lab)
    
    R = len(classes)

    f = numpy.zeros((1,R), dtype='f')

    for r in range(R):

        v = numpy.nonzero(classes == lab)[0]
        
        f[r] = norm(alpha[v], ord=1) / norm(alpha, ord=1)

    return (R*f.max() - 1) / (R-1)
