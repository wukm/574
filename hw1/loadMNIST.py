#!/usr/bin/env python3

#   TODO
#   -   needs heavy refactoring for readability (I MADE A SPAGHETTI)
#   -   whatever the **** is going on in loadMNIST.m: lines 23-25
#   -   match interface of loadMNIST.m or ... don't?
#   -   digits is potentially a list
#   -   these functions are supposed to return labels
#   -   portability

"""
loadMNIST.py
author: luke wukmer

NOTE: This is actually the entirety of the HW1 q5 code, not just a porting of
loadMNIST.m. Hopefully a cleaned up version of this will surface later (this
code is also hosted on github: the repository is at github.com/wukm/574).
Also, I apologize for the disregard to portability, I may work on that at a
later date.
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

def load_all(digit):
    """
    returns an A matrix for both training and testing images for a particular
    digit (at this point, digit is a scalar only). Note this is equivalent to
    the provided matlab function load_Images( ... , 'all')

    returns a (n1 + n2) by (28*28) matrix, where n1 is the number of
    testing images and n2 is the number of training images.

    it is concatenated predicably: the first n1 are from 'testing', and the next
    n2 are from 'training'
    """

    tests = load(digit, type_str='test')
    trains = load(digit, type_str='train') # choo choo
    
    return numpy.concatenate((tests, trains), axis=0)

def load(digit, type_str='train'):
    """
    this returns a matrix whose columns are the images corresponding to the
    digit `digit`.

    resulting dimension of outputted matrix will be
        (number of hits in image set, 28*28)
    """
    assert type_str in ('test', 'train'), "use the load_all function"

    if type_str == 'test':
        base = 'testing'
    else:
        base = 'training'

    img_file = '{}_images'.format(base)
    lbl_file = '{}_labels'.format(base)
    images = load_images(img_file)
    labels = load_labels(lbl_file)

    # make a list of the rows that correspond to `digit`
    relevant = [images[i] for i , label in enumerate(labels) if label == digit]

    return numpy.array(relevant)

def rank_two_approx(A):
    """ returns the approximated matrix, as well as the u, s, v of the
    decomposition as well. a minor bother is that the *second* singular value is
    actually greater than the first -- it seems that scipy.sparse.svds outputs
    in opposite order. doesn't matter.
    """
    u, s, v = sla.svds(A, k=2, which='LM')
    
    # nasty syntax
    approx = u.dot(s*numpy.eye(2)).dot(v)
    return approx, (u, s, v)

    
def load_images(filename='training_images'):
    """extracts images from the binary blobs "*_images". """ 
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, 'rb') as f:
        b = f.read() # hope ya get it all

    # grab the first four numbers ...
    # fmt='>i' means big-endian int32
    magic_no, n_images, n_rows, n_cols = (struct.unpack('>i', b[i*4:(i+1)*4]) for i in range(4))

    # i am a god-fearing man
    assert magic_no[0] == 2051, "bad magic number, what do?"


    # so i think you can use the standard libary's "array" for this, just
    # because binary data of any sort is kinda dodgy, but this grabs 'the rest'
    # format='B' means unsigned char === 'uint8', and apparently endianness doesn't matter
    image_stream = array.array('B', b[16:])

    # so each 28*28 byte portion of image_stream is a flattened image. these two
    # numpy.reshape calls get it into the desired shape for A. maybe could
    # combine it into one call, idk. anyway, each flattened image appears as a
    # row, and there is a row for each image.
    image_first = numpy.reshape(image_stream, (n_images[0], n_rows[0], n_cols[0]))
    images = image_first.reshape(n_images[0], n_rows[0]*n_cols[0])

    # convert to float in [0,1]
    images = images.astype('f') / 255

    return images 
    
def load_labels(filename):
    """ very similar idea to load_images(...), see above.
        but in this case, returns a tuple."""

    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, 'rb') as f:
        b = f.read()

    magic_no, n_labels = (struct.unpack('>i', b[i*4:(i+1)*4]) for i in range(2))

    assert magic_no[0] == 2049, "bad magic number, what do?"

    label_stream = array.array('B', b[8:])
    
    assert len(label_stream) == n_labels[0], "mismatch in label length"
    
    # label_stream is actually type array.array, which is iterable surely.
    # i'll convert it anyway...
    return tuple(label_stream)

if __name__ == "__main__":

    assert os.path.isdir(DATA_DIR), "cannot find data directory in {}".format(DATA_DIR)
        
    for d in (1,2):
        A = load_all(d)
        AA, (u, s, v) = rank_two_approx(A)
       
        # convert right singular vectors to image shape
        v1 = v[0].reshape((28,28))
        v2 = v[1].reshape((28,28))

        # because of dumbness in the interface, v2 is actually the right
        # singular vector corresponding to the largest singular value
        v2, v1 = v1, v2

        v1 = normalize_grayscale(v1)
        v2 = normalize_grayscale(v2)
        
        # convert to uint8 again for easier writing
        v1 = v1.astype('uint8')
        v2 = v2.astype('uint8')
        
        file_base = "all_{}_{}.tiff"
        Image.fromarray(v1).save(file_base.format(d, 'v1'))
        Image.fromarray(v2).save(file_base.format(d, 'v2'))
        
        # see comment above
        u1 = u.T[1]
        u2 = u.T[0]

        # returns index of maximum / minimum in question
        ibig, ismall = u2.argmax(), u1.argmin()
        
        big = A[ibig].reshape((28,28))
        small = A[ismall].reshape((28,28))

        big = normalize_grayscale(big)
        small = normalize_grayscale(small)

        big = big.astype('uint8')
        small = small.astype('uint8')

        Image.fromarray(big).save(file_base.format(d, 'big'))
        Image.fromarray(small).save(file_base.format(d, 'small'))



