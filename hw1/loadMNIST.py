#!/usr/bin/env python3

#   TODO
#   -   I MADE SPAGHETTI
#   -   needs heavy refactoring for readability
#   -   needs to accept input of 'training' 'testing' whatever
#   -   whatever the hell is going on in loadMNIST.m lines 23-25
#   -   match interface of loadMNIST.m or ... don't?

"""
loadMNIST.py

here's the spec i guess:

INPUT
    digits, type.

digits -> int(s) that specify which digits to load, rather than all of them... i
            think? 
type -> ['test', 'train', 'all']
    which has to do with how much crap to return.

# OUTPUT
    images, labels

images -> i guess... an array of pixel matrices, in 256-bit color
labels -> idk

"""

import struct
import array
import numpy
import os
import scipy.sparse.linalg as sla
from PIL import Image
from opentiff import normalize_grayscale

# this is one step better than hardcoding but still sloppy
DATA_DIR = os.path.join(os.getcwd(), 'data/', 'MNIST/')

def load(digit, type_str='all'):
    """
    this returns a matrix whose columns are the images corresponding to the
    digit `digit`. and yes, right now there can only be one digit.
    fix that later.
    """
    #assert type_str in ('all', 'test', 'train'), "need a valid label"
    #assert type_str == 'test', "i've only coded the part for test" 

    images = load_images('testing_images')
    labels = load_labels('testing_labels')  
    
    #images = images.T

    # make a list of the columns that correspond to `digit`
    relevant = [images[i] for i , label in enumerate(labels) if label == digit]

    # resulting dimension will be (28*28, number of hits in image set)
    return numpy.array(relevant)

def rank_two_approx(A):
    u, s, v = sla.svds(A, k=2, which='LM')
    approx = u.dot(s*numpy.eye(2)).dot(v)
    return approx, (u, s, v)

    
def load_images(filename='training_images'):

    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, 'rb') as f:
        b = f.read() # hope ya get it all

    # grab the first four numbers ...
    # fmt='>i' means big-endian int32
    magic_no, n_images, n_rows, n_cols = (struct.unpack('>i', b[i*4:(i+1)*4]) for i in range(4))

    assert magic_no[0] == 2051, "bad magic number, what do?"

    # format='B' means unsigned char, and apparently endianness doesn't matter?

    # so i think you can use the standard libary's "array" for this?
    # struct said it didn't actually support type int, meaning it was already
    # int? idk man.
    image_stream = array.array('B', b[16:])

    # okay, so now the hard part is knowing how to translate this:
    # images = reshape(images, numCols, numRows, numImages)
    # images = permute(images, [2,1,3])

    # ugh, i forgot struct returned tuples for fun.
    image_first = numpy.reshape(image_stream, (n_images[0], n_cols[0], n_rows[0]))
    #images = image_first.swapaxes(2,3)
    images = image_first.reshape(n_images[0], n_rows[0]*n_cols[0])
    #images = images.reshape(n_rows[0]*n_cols[0], n_images[0])
    images = images.astype('f') / 255
    return images 
    # each image appear as a row vector in images. the -1 means 'infer this'
    # images.shape -> (28*28, n_images)
    #images = numpy.array((image_stream).reshape((28*28,-1))
    #return images
    #images = images.astype('float32')
    #images = images / 255.
    #return images.T
    
def load_labels(filename):
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, 'rb') as f:
        b = f.read()

    magic_no, n_labels = (struct.unpack('>i', b[i*4:(i+1)*4]) for i in range(2))
    20 

    assert magic_no[0] == 2049, "bad magic number, what do?"
    label_stream = array.array('B', b[8:])
    
    assert len(label_stream) == n_labels[0], "mismatch in label length"
    
    # label_stream is actually type array.array, does it matter?
    # i'll convert it anyway...
    return tuple(label_stream) 

if __name__ == "__main__":

    assert os.path.isdir(DATA_DIR), "cannot find data directory in {}".format(DATA_DIR)
    
    for d in (1,2):
        A = load(d)
        AA, (u, s, v) = rank_two_approx(A)

        v1 = v[0].reshape((28,28))
        v2 = v[1].reshape((28,28))
        
        v1 = normalize_grayscale(v1)
        v2 = normalize_grayscale(v2)

        Image.fromarray(v1).show()
        Image.fromarray(v2).show()
