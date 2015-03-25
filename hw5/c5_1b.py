#!/usr/bin/env python3

import numpy
from kernel import gaussian_N, gaussian_M, as_column_vector
from kernelSVM import kernel_classify
from loadISOLET import load_isolet

if __name__ == "__main__":


    vowels = [1, 5, 9, 15, 21]

    voice, lab = load_isolet(vowels, 'train')
    voice_t, lab_t = load_isolet(vowels, 'test')
    
    C = .01
    sigma = 9

    # each takes about 4 seconds
    M = gaussian_M(voice, sigma)
    N = gaussian_N(voice_t, voice, sigma)
   
    # one vs all
    
    F = list()
    F_act = list()

    def _get_y(vi):
        y = 2 * (lab == vi) - 1 
        return as_column_vector(lab == vi) 
   
    # might as well do a for loop
    for vi in vowels:
        
        y = _get_y(vi)
        gamma = kernel_classify(M, y, C)

        # please package this sloppy shit up already
        supp = (0 < abs(gamma)) & (abs(gamma) < C)
        s = numpy.nonzero(supp)[0]
        v = M.dot(gamma)
        # ~*~*tada*~*~
        alpha = (y[s] - v[s]).sum() / s.size

        F.append(alpha + N.dot(gamma))
        F_act.append(lab_t == vi)

        
        break 
