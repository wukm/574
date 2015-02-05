#!/usr/bin/env python3

#   TODO
#   -   needs heavy refactoring for readability
#   -   needs to accept input of 'training' 'testing' whatever
#   -   whatever the hell is going on in loadMNIST.m lines 23-25
#   -   match interface of loadMNIST.m or ... don't?

"""
loadMNIST.py

here's the spec i guess:

INPUT
    digits, type.

digits -> ???
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

def load_images(filename='data/MNIST/training_images'):
    with open(filename, 'rb') as f:
        b = f.read() # hope ya get it all

    # grab the first four numbers ...
    # fmt='>i' means big-endian int32
    magic_no, n_images, n_rows, n_cols = (struct.unpack('>i', b[i*4:(i+1)*4]) for i in range(4))

    assert magic_no[0] == 2051, "HOLY SHIT. RUN"

    # format='B' means unsigned char, and apparently endianness doesn't matter?

    # so i think you can use the standard libary's "array" for this?
    # struct said it didn't actually support type int, meaning it was already
    # int?
    # idk man
    image_stream = array.array('B', b[16:])

    # okay, so now the hard part is knowing how to translate this:
    # images = reshape(images, numCols, numRows, numImages)
    # images = permute(images, [2,1,3])

    # just hoping the numpy.reshape interface is the same...
    # also, ugh, i forgot struct returned tuples for fun.
    image_first = numpy.reshape(image_stream, (n_images[0], n_cols[0], n_rows[0]))

    # now to imitate the permute command...
    images = image_first.swapaxes(1,2)

    # and this other wonky shit he did
    images = images.reshape(n_rows[0]*n_cols[0], n_images[0])
    images = images.astype('f') / 255

    return images

def load_labels(filename='data/MNIST/training_labels'):
    with open(filename, 'rb') as f:
        b = f.read()

    magic_no, n_labels = (struct.unpack('>i', b[i*4:(i+1)*4]) for i in range(2))
    20 

    assert magic_no[0] == 2049, "bad magic number: {}".format(magic_no[0])
    label_stream = array.array('B', b[8:])
    
    assert len(label_stream) == n_labels, "Mismatch in label count"
    
    # label_stream is actually type array.array, does it matter?
    # i'll convert it anyway...
    return tuple(label_stream) 
    return label_stream
