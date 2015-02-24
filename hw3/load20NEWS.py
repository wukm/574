#!/usr/bin/env python3

## TODO
##  -   again, filenames are hardcoded. fix
##  -   just pickle the datasets in matrix form. way easier     
##  -   correct output
##  -   finish switch

"""
load20NEWS.py

This is a port of the provided load20NEWS.m
"""

import numpy
import from scipy import sparse

def _int_array_from_file(filename, dtype=None, sep=None):
    """
    this is supposed to mimic dlmread in MATLAB

    the following alternatives were way too slow. mostly separating and type
    converting seem to be taking forever for a large file
    -   numpy.loadtxt(filename, dtype='uint32')
        (although this makes the correct shape right away)
    -   csv
    -   any string manipulations
    -   array.array
        (absolutely wrong, can't handle delimiters)

    alternative is to just do it the 'right way' (which seems to be
    numpy.loadtxt), and then pickle the output.
    """

    if dtype is None:
        dtype = 'uint32'

    if sep is None:
        sep = ' '

    with open(filename) as f:
        # this can only handle one kind of separator, so it will return a flat
        # array. but it's very fast in comparison.
        flat = numpy.fromfile(f, dtype=dtype, sep=sep)

        # now awkwardly get the shape. only need one dimension.
        f.seek(0)
        cols = len(f.readline().split(sep))

    # will be in the same orientation as the file now.
    return flat.reshape((-1, cols))

def load_20_news(groups, subset):
    """
    Parameters
    ----------

    groups: a sequence of integers 1-20, specifying which groups of 'articles'
            to load

    subset: a label. must be in ('test', 'train', 'all'), which sets of data to
            pull these 'articles' from

    Returns
    -------

    docs    data set

    lab:

    vocab:
    """
    # i really hate these switch statements, but what do?
    if subset == 'test':
        docs = _int_array_from_file('test.data') 
        lab = _int_array_from_file('test.label')  
    elif subset == 'train':
        docs = _int_array_from_file('train.data') 
        lab = _int_array_from_file('train.label')
    elif subset == 'all': 
        elif subset == 'all':
        X1 = _int_array_from_file('test.data') 
        X2 = _int_array_from_file('test.data') 
        T1 = _int_array_from_file('train.label')  
        T2 = _int_array_from_file('train.label')  

        # the first column of each of the 'X' sets is an index.
        # this just offsets the index in X2 by the size of X1 so that the
        # combined X is all in order. pretty useless but I'll play along.
        X2[:,0] += len(T1)

        docs = numpy.concatenate((X1,X2), axis=0)
        lab = numpy.concatenate((T1,T2), axis=0)
    else
        raise Exception("subset must be 'train', 'test', or 'all'")

    # now separate columns
    iv, jv, vv = docs.T
    
    # literally just copying matlab here :/ csc_matrix is one of apparently
    # 7 ways that scipy.sparse can create a sparse matrix; this is the one
    # matlab uses. not sure why we're getting the shape this way.
    #docs = sparse(iv,jv,vv,max(iv),61188);
    docs = sparse.csc_matrix((vv, (iv,jv)), shape=(iv.max(), 61188))

    #idx = bsxfun(@eq,lab,groups);
    #docs = docs( sum(idx,2) == 1 , :);
    #lab = lab( sum(idx,2) == 1 );
    #
    
    # BUILD VOCAB
    with open('_vocabulary.txt') as f:
        vocab = [line.strip() for line in f] 

    # what does this line do? resolve dim?
    #vocab = vocab{1};
        
    return docs, lab, vocab
