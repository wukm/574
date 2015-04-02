#!/usr/bin/env python3

import numpy

def gamma_to_hyperplane(gamma, X, y, C):
    """
    given a gamma (obtained via SVM)
    find the coefficients α, β of the separating hyperplane

    α here is found via an average. other ways to do this of course

    needs the initial inputs (X, y, C) to the dual problem of course

    """
    
    beta = numpy.dot(X.T, gamma)

    # get a boolean array, same shape as gamma
    # true when 0 < |γ_i| < C
    supp = (0 < abs(gamma)) & (abs(gamma) < C)
    s = numpy.nonzero(supp)[0]

    alpha = (y[s] - X[s].dot(beta)).sum() / s.size

    return alpha, beta
    

