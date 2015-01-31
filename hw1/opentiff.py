#!/usr/bin/python3

# this shit opens tiff files. trying to figure out how this works, get it into a
# matrix of pixels.

# so apparently these are 256 color images.
from PIL import Image
import numpy as np
import scipy.sparse.linalg as sla

# please use os or something for file manipulations; this is really ugly.
# hope you're in the right directory you sad shit ;)
image_files = [''.join(('data/MOTION/motion_',str(n),'.tiff')) for n in range(1,11)]


# note that np.array() can accept either im by itself (which it converts to a
# matrix), or im.getdata() to immediately columnize it. prolly a bunch of other
# ways too.
vectors = []

# steps to avoid i bet. idk. can you build an np.array with a generator?

for image in image_files:
    im = Image.open(image)
    vectors.append(np.array(im.getdata()))

A = np.array(vectors, dtype='f').T

# k=1 only compute 1 singular value
# which='LM' means only compute largest
# note that v is actually v^T
u, s, v = sla.svds(A, k=1, which='LM')

# s is still in array form. not sure what's going on but this is one way
s1 = s.item()

# then this is the rank 1 approximation of A
rank_one = s1*u*v

fg_images = A - rank_one

backgrounds = []
foregrounds = []
# vectors already filled

# iterate over columns
for column in fg_images.T:
    # make a picture
    temp = Image.fromarray(column.reshape((512,512)))
    foregrounds.append(temp) 

for column in rank_one.T:
    temp = Image.fromarray(column.reshape((512,512)))
    backgrounds.append(temp)

for i, image in enumerate(foregrounds):
    filestring = 'data/MOTION/fg_{}.gif'.format(i+1)
    image.save(filestring)

for i, image in enumerate(backgrounds):
    filestring = 'data/MOTION/bg_{}.gif'.format(i+1)
    image.save(filestring)
