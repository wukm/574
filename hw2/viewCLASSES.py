#!/usr/bin/env python3
"""
loadCLASSES.py

ported from loadCLASSES.m
provides a method to examine fg/bg extraction with a viewable image
"""

import numpy
import PIL.Image

def make_8bit_array(array):
    """
    converts a matrix with float elements between 0 and 1
    to one with int elements in (0,255)
    """
    return (255 * array).astype('int')
def view_classes(img,y):
    """
    Input:
    img -> a matrix corresponding to an image. size is...?
    y -> a column vector of classification of each pixel in img

    Output:
    (nothing)

    This calls the method PIL.Image.show(...) on the fg extraction and
    background extraction of the img.

    """

    # tracing dimensionality through the original matlab code
    # input is img which is an 400X512X3 matrix
    # y is a m*n vector (row i guess...)

    dim = img.shape[:2]
    print(dim) 
    forg = numpy.array(y > .5, dtype='d')
    back = numpy.array(y <= .5, dtype='d')

    for i, mat in enumerate((forg, back)):
        print(mat.shape)
        mat = mat.reshape(dim)
        mat = numpy.expand_dims(mat, 2)
        mat = numpy.tile(mat, (1,1,3)) # might need to be (3,1,1)?
        assert mat.shape == img.shape, "you messed up the dimensions"
        im = mat * img # should be same element size
        im = make_8bit_array(im)
        
        #PIL.Image.fromarray(im, mode='RGB').show(title="figure {}".format(i+1))
    PIL.Image.fromarray(img, mode='RGB').show()
    return None
