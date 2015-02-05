#!/usr/bin/python3

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

def rank_one_approx(A):
    """
    computes a rank one approximation of the matrix A
    with usual shit error handling.
    returns a matrix the same shape as A 
    """
    # k=1 only compute 1 singular value
    # which='LM' means only compute largest
    # note that v is actually v^T in the class notes.
    u, s, v = sla.svds(A, k=1, which='LM')

    # s is still in array form by default. 
    s1 = s.item()

    # should actually check stuff like same shape maybe? idk
    approx = s1*u*v
    return approx

rank_one = rank_one_approx(A)
foregrounds = A - rank_one

def normalize_grayscale(A):
    """return a linearly normalized copy of the matrix with
       0 as min and 1 as max.
       note the matrix A must be float type of some sort.
       i actually don't know what happens in memory vs. performing
       these operations in the global namespace :/

       also note: returns same type. integer type is okay, but isn't really
       necessary for passing to PIL.Image(...)
    """
    # must be able to handle float point arithmetic, but what do?
    # assert A.dtype == np.int32 
    A_max, A_min = (A.max(), A.min())
    if A_max - A_min == 0:
        # should be descriptive later
        raise
        return A
    else:
        return 255 * (A - A_min) / (A_max - A_min)
bg_matrices = [linear_normalize(column) for column in rank_one.T]
# vectors already filled

# iterate over columns
#for column in fg_images.T:
#    # make a picture
#    temp = Image.fromarray(column.reshape((512,512)))
#    foregrounds.append(temp) 
#    fga.append(column.reshape((512,512)))
#for column in rank_one.T:
#    temp = Image.fromarray(column.reshape((512,512)))
#    backgrounds.append(temp)
#
#for i, image in enumerate(foregrounds):
#    filestring = 'data/MOTION/fg_{}.gif'.format(i+1)
#    image.save(filestring)
#
#for i, image in enumerate(backgrounds):
#    filestring = 'data/MOTION/bg_{}.gif'.format(i+1)
#    image.save(filestring)
