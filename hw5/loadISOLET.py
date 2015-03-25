#!/usr/bin/env python3

"""
loadISOLET.py
basically the same structure as loadMNIST.py i think
"""

import numpy


def load_isolet(letters, typestr):
    """
    input:
        letters: an iterable within components in {1, 2, ..., 26}
        typestr: 'test' 'train' or 'all'
    output:
        voice (data matrix, float)
        lab (class labels), float, 2D

    load the ISOLET dataset, found at https://archive.ics.uci.edu/ml/datasets/ISOLET
    although i think my input files are a slightly different format than what's
    found there.

    Note: This takes about 4 seconds to load (regardless of what 'letters' is,
    it seems) when loading 'all', so not super necessary to pickle. still
    consider if the main loop runs long.
    
    """
    
    # again, hardcoded filenames are bad
    
    if typestr == 'test':
        X = numpy.loadtxt('data/ISOLET/isolet.test', dtype='d', delimiter=', ')

    elif typestr == 'train':
        X = numpy.loadtxt('data/ISOLET/isolet.training', dtype='d', delimiter=', ')
    elif typestr == 'all':
        X1 = numpy.loadtxt('data/ISOLET/isolet.training', dtype='d', delimiter=', ')
        X2 = numpy.loadtxt('data/ISOLET/isolet.test', dtype='d', delimiter=', ')
        X = numpy.vstack((X1, X2))
    
    voice, lab = X[:,:-1], X[:,-1]

    # make a boolean index array to pare down in letters isn't all.
    # can't use `in` directly, also can't be a list. fiik
    # this is fast enough though
    bi = numpy.array([label in letters for label in lab]) 

     
    
    # note lab has shape (7797,) (in the case of 'all')
    # so i'm gonna reshape to a 2d array.
    return voice[bi], lab[bi].reshape((-1,1))
