#!/usr/bin/env python3
"""
loadCLASSES.py

ported from loadCLASSES.m
provides a method to examine fg/bg extraction with a viewable image
"""

import numpy

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
    m, n = img.shape

    #forg = numpy.astype(y > .5, 'double')
    return None

