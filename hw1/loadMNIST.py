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
code is also hosted on github: the repository is at github.com/wukm/579).
Also, I apologize for the disregard to portability, I may work on that at a
later date.
"""

import struct
import array
import numpy
import os

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

    
def load_images(filename='training_images'):
    """extracts images from the binary blobs "*_images". """ 
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, 'rb') as f:
        b = f.read() # hope ya get it all

    # grab the first four numbers ...
    # fmt='>i' means big-endian int32
    magic, n_images, n_rows, n_cols = (struct.unpack('>i', b[i*4:(i+1)*4]) for i in range(4))

    # i am a god-fearing man
    assert magic[0] == 2051, "bad magic number, what do?"


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

    magic, n_labels = (struct.unpack('>i', b[i*4:(i+1)*4]) for i in range(2))

    assert magic[0] == 2049, "bad magic number, what do?"

    label_stream = array.array('B', b[8:])
    
    assert len(label_stream) == n_labels[0], "mismatch in label length"
    
    # label_stream is actually type array.array, which is iterable surely.
    # i'll convert it anyway...
    return tuple(label_stream)

