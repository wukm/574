#!/usr/bin/env python3

"""
solution to coding question 2 from homework 3
"""
from load20NEWS import load_20_news
from LOGClassify2 import log_energy, log_classify
import numpy

# part (a)
groups = (10,11)
docs, lab, vocab = load_20_news(groups, 'train')
docs_t, lab_t, _ = load_20_news(groups, 'test')

# make y
y = numpy.array(lab == 10, dtype='double')

# part (b)

dt=.00001

lambdapickles = ((1.0, '20news_1.pickle'),
                    (10., '20news_10.pickle'),
                    (100., '20news_100.pickle'),
                    )
energy_lines = []
accuracy_lines = []
for lamb, picklefile in lambdapickles:

    try:
        *beta, alpha = numpy.load(picklefile)
    except FileNotFoundError:
        print("no pickle :(")
        # the pickle doesn't exist, so make it
        alpha, beta = log_classify(docs, y, lamb, dt0=dt)
        to_pickle = list(beta) + [alpha]
        to_pickle = numpy.array(to_pickle)
        to_pickle.dump(picklefile)

    beta = numpy.array(beta)

    E = log_energy(alpha,beta, docs, y, lamb)
    energy_line = "final energy with λ={} was E={}\n".format(lamb,float(E))
    energy_lines.append('-'*80 + '\n')
    energy_lines.append('λ = {}'.format(lamb) + '\n')
    energy_lines.append(energy_line)
    # part c
    binary_est = (alpha + docs_t * beta) > 0

    actual = tuple(test_label == groups[0] for test_label in lab_t)
    computed = tuple(binary_est)
    
    assert len(actual) == len(computed)

    hits = [x == y for x,y in zip(actual,computed)]
    total_hits = hits.count(True)
    
    m = (total_hits / len(actual)) * 100
    accuracy_line = 'accuracy with λ={} was {}/{} = {}\n\n'.format(lamb,
            total_hits, len(actual), m)
    #print(accuracy_line)
    energy_lines.append(accuracy_line)

    b = beta.flatten()
    indices = b.argsort()
    class0_words = ' '.join(vocab[indices[:10]])
    class1_words = ' '.join(vocab[indices[-10:]])
    energy_lines.append('\n'.join(("words found from group {}:".format(groups[0]), class0_words)))
    energy_lines.append('\n\n')
    energy_lines.append('\n'.join(("words found from group {}:".format(groups[1]), class1_words)))
    energy_lines.append('\n'+'-'*80 + '\n') # wow

with open('c3_2.txt', 'w') as f:
    f.writelines(energy_lines)
    #f.write('\n')
    f.writelines(accuracy_lines)
