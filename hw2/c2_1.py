#!/usr/bin/env python3

from loadCOLOR import load_color
from LSclassify import ls_classify
from viewCLASSES import view_classes

import numpy

img, X, Xtrain, y = load_color()

def classify(x, α, β):
    """
    classify(x, α, β) -> y in {0,1}

    returns the label y for the given data point x by
        y = α + < β , x >

    OOPS I DON'T NEED TO THIS, I'LL JUST WORK ON THEM AS MATRICES
    """
    pass
#planes = tuple((λ, ls_classify(Xtrain, y, λ)) for λ in (.005, .5, 10, 250, 800))

λ = .005
α, β = ls_classify(Xtrain, y, λ)

Y = α + numpy.dot(X, β)

view_classes(img, Y)
