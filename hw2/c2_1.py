#!/usr/bin/env python3

"""
c2_1.py

(Coding I) / (HW 2 q4)

finds (α, β) via least squares approximation

and classifies the pixels X via 
    y = α + < β , x >

then view_classes() is called to display them
"""

from loadCOLOR import load_color
from LSclassify import ls_classify
from viewCLASSES import view_classes

import numpy

def main():

    img, X, Xtrain, y = load_color()

    for λ in (.005, .5, 10, 250, 800):
        α, β = ls_classify(Xtrain, y, λ)
        Y = α + numpy.dot(X, β)

        view_classes(img, Y, λ)


if "__name__" == "__main__":

    main()
