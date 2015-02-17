#!/usr/bin/env python3

#   TODO
#   -   needs heavy refactoring for readability (I MADE A SPAGHETTI)
#   -   match interface of loadMNIST.m or ... don't?
#   -   these functions are supposed to return labels
#   -   make top level load_mnist() function
#   -   if above done, for the love of god read the files once only per
#       function call, not (up to) four times.
#   -   portability

"""
loadMNIST.py
author: luke wukmer

ohh, so this is exactly the data set i'm working with:
http://en.wikipedia.org/wiki/MNIST_database

...that explains the cryptic name
"""

import struct
import array
import numpy
import os

# this is one step better than hardcoding but still sloppy
DATA_DIR = os.path.join(os.getcwd(), 'data/', 'MNIST/')

def load_mnist(digits, subset='all', path=None):
    """
    load_mnist(digits, [subset='all' [,path=None]])) -> images, labels

    INPUT:

    digits is an iterable of ints 0-9 of digits to return. (repetitions are
    tolerated).

    if specified, subset must be 'all' 'train' or 'test'. this will load a
    particular subset of the MNIST data set. default is 'all'

    if specified, path is a string which specifies which directory the files are
    in. the following files are

    (path)
    ├── testing_images
    ├── testing_labels
    ├── training_images
    ├── training_labels
    └── (...)

    if path is not specified, the function searches the result of

        os.path.join(os.getcwd(), 'data/', 'MNIST/')

    although this behavior is subject to change.

    OUTPUT:

    `images` is a N by (28*28) numpy.array, where N is the total number of
    images requested, with the 28*28 pixels of each image vectorized.
    
    `labels` is a N-tuple, where N is the total number of images requested.

    n.b. This (should be) the only public function here.
    """ 
    
    if path is None:
        path = os.path.join(os.getcwd(), 'data/', 'MNIST/')

    # should fix this. so much redundancy. files can be read up to 4 times
    if subset == 'all':
        images, labels = load_all(digits)
    elif subset in ('train', 'test'):
        images, labels = load(digits, subset)
    else:
        #do better here
        raise Exception('please specify a valid subset')
    
    return images, labels

def load_all(digits):
    """
    returns an A matrix for both training and testing images for a particular
    digit (at this point, digit is a scalar only). Note this is equivalent to
    the provided matlab function load_Images( ... , 'all')

    returns a (n1 + n2) by (28*28) matrix, where n1 is the number of
    testing images and n2 is the number of training images.

    it is concatenated predicably: the first n1 are from 'testing', and the next
    n2 are from 'training'
    """

    tests, test_labels = load(digits, type_str='test')
    trains, train_labels = load(digits, type_str='train') # choo choo

    images = numpy.concatenate((tests, trains), axis=0)
    labels = test_labels + train_labels 

    return images, labels

def load(digits, type_str='train'):
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

    # list of tuples ( (row corresponding to a relevant digit), (its label) )
    # probably a simpler way to do this, eh. 
    relevant = [(images[i] , label) for i , label in enumerate(labels) if label in digits]
    images = numpy.array([x[0] for x in relevant])
    labels = tuple(x[1] for x in relevant)
    
    return images, labels

    
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


   
    # using array from stdlib, since binary data is dodgy.
    # honestly int(b[16:]) works in practice format='B' means unsigned char ===
    # 'uint8', and apparently endianness doesn't matter
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
    
    return tuple(label_stream)

