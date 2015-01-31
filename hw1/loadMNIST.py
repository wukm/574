#!/usr/bin/env python3

# TODO: get labels doing whatever they're supposed to.
#       make this code not look like crap

"""
loadMNIST.py

holy shit, i get to load some images (i hope) from a fucking binary blob.
JVB provided the source for this magic function of making the blob useful, but
it was written in MATLAB. So yeah, here's to fucking reverse engineering

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

#def parse_training_images(filename='./data/MOTION/MNIST/training_images'):
with open('data/MNIST/training_images', 'rb') as f:
    b = f.read() # hope ya get it all

# grab the first four numbers ...
# fmt='>i' means big-endian int32
magic_no, n_images, n_rows, n_cols = (struct.unpack('>i', b[i*4:(i+1)*4]) for i in range(4))

#assert magic_no == 2051, "HOLY SHIT. RUN"

# format='B' means unsigned char, and apparently endianness doesn't matter?

# so i think you can use the standard libary's "array" for this?
# struct said it didn't actually support type int, meaning it was already int?
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
