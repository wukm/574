#/usr/bin/env python3

"""
loadCOLOR.py

This is a port of the provided MATLAB function loadCOLOR.m

accepts no inputs (really there are several files it loads)
output is as follows:

img ->      a 400 x 512 x 3 matrix (like a set of three 400 by 512 matrices)
        with elements between 0 and 1.
X   ->      a redundant version of `img` above; it is a (400*512) by 3 matrix.
        essentially the 400x512 matrices are now row vectors.
Xtrain ->   the exact contents of X reduced to only the 'training' portion of
        the data. specifically, it has 36670 rows, where the first 29850 rows
        correspond to the foreground, and the remainder correspond to the
        background.
y   ->      a classification vector for the training data. the i-th component of
        y is the classification (1 -> fg , 0 -> bg) of the i-th pixel in Xtrain
        above.
"""
