#!/usr/bin/env python3

"""
c2_2.py

(Coding II) / (HW 2 q5)

a short description would be here... :)
"""
from LSclassify import ls_classify
from loadMNIST import load_mnist
from scipy.stats import mode
import numpy


def accuracy(computed, actual, num_classes):
    """
    port of ACCURACY.m
    could clean this up
    """

    n = len(actual)

    modes = numpy.zeros(num_classes,1)
    for k in range(num_classes):
        v = actual(computed == k)
        modes[k] = mode(v)[1] 

    return (sum(modes) / n) * 100

def lsq_on_mnist(digits):
    """
    lsq_on_mnist( digits ) -> m

    digits -> digits to use
    outputs the accuracy of the approximation
    """

    assert len(digits) == 2, "binary classification only (for now?)"

    c = digits[1] # the digit that will correspond to class 1

    # load training pix
    X_train, labels = load_mnist(digits, 'train')

    y = numpy.array([1. if label == c else 0. for label in labels])

    λ = 20.0 # use this for no reason

    α, β = ls_classify(X_train, y, λ)

    X, test_labels = load_mnist(digits, 'test')

    #binary  = numpy.array([1. if label == 6 else 0. for label in test_labels])

    # sorry, rolling my own for the rest of this since it's just 2D

    binary_est = (α + X.dot(β)) > 0.5

    actual = tuple(label == c for label in test_labels)
    computed = tuple(binary_est)
    
    assert len(actual) == len(computed)

    # how many times the computed label matched the actual label
    hits = [x == y for x,y in zip(actual,computed)]
    total_hits = hits.count(True)
     

    m = (total_hits / len(actual))* 100 #total % accuracy

    # could also return images that were unmatched, eh?
    return total_hits, len(actual), m

if __name__ == "__main__":
    # here's the actual question
    
    for digits in [(1,6), (4,9)]:
    
        print('for digits={}'.format(digits))
    
        hits, total, percent = lsq_on_mnist(digits)

        print('{} hits out of {}, {}%'.format(hits,total,percent)) 
