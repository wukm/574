#!/usr/bin/env python3

from loadCOLOR import load_color
from LOGClassify import log_energy, log_classify
# could really just use pickle here instead of numpy
import numpy
from viewCLASSES import view_classes

img, X, Xtrain, y = load_color()

lambdapickles = [ ( .005,'color_005.pickle'),
                (.05, 'color_05.pickle'),
                (.5, 'color_5.pickle'),
                ]
energy_lines = []

for lamb, picklefile in lambdapickles:

    try:
        *beta, alpha = numpy.load(picklefile)
    except FileNotFoundError:
        print("no pickle :(")
        # the pickle doesn't exist, so make it
        alpha, beta = log_classify(Xtrain, y, lamb)
        to_pickle = list(beta) + [alpha]
        to_pickle = numpy.array(to_pickle)
        to_pickle.dump(picklefile)
        print("made pickle")

    beta = numpy.array(beta)
    E = log_energy(alpha, beta, Xtrain, y, lamb)
    energy_lines.append('λ={}\t'.format(lamb))
    #print('found α: {}'.format(alpha))
    #print('found β: {}'.format(beta))
    energy_lines.append("final energy is {}\n".format(E))
   
    Y = alpha + numpy.dot(X,beta)
    view_classes(img,Y,lamb,save=True)

with open('c3_1.txt', 'w') as f:
    f.writelines(energy_lines)
