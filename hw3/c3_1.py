#!/usr/bin/env python3

from loadCOLOR import load_color
from LOGClassify import log_energy, log_classify

img, X, Xtrain, y = load_color()

lamb = .005
alpha, beta = log_classify(Xtrain, y, lamb)

