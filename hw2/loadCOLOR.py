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

import PIL.Image
import numpy

def make_float_array(filename):
    """
    Input:  filename of 256-color (grayscale or RGB) image
    Output: a numpy.array with dtype float (scaled between 0 and 1)

    Gross function name, whatever.
    Dimension is not altered in any way.
    """
    
    im_raw = PIL.Image.open(filename)
    im = numpy.array(im_raw, dtype='f')
    im /= 255
    return im

#   should not be raw file paths, that's ugly.

img = make_float_array('data/COLOR/image.jpg')
forg_mask = make_float_array('data/COLOR/mask0.jpg')
back_mask = make_float_array('data/COLOR/mask1.jpg')

# surely there's a better way to do this?
# vectorize each and make a new matrix with those vectors as rows

dim = (img.shape[0]*img.shape[1], 1)
R = img[:,:,0].reshape(dim)
G = img[:,:,1].reshape(dim)
B = img[:,:,2].reshape(dim)

X = numpy.concatenate((R,G,B),1)

forg = numpy.sum(forg_mask, 2)
forg = forg.reshape(dim)
forg_pix = numpy.nonzero(forg > 0.5)
