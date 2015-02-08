#!/usr/bin/python3

from PIL import Image
import numpy as np
import scipy.sparse.linalg as sla

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

def normalize_grayscale(A):
    """return a linearly normalized copy of the matrix with
       0 as min and 1 as max.
       note the matrix A must be float type of some sort.
    """
    # must be able to handle float point arithmetic, but what do?
    #assert A.dtype == np.int32 
    A_max, A_min = (A.max(), A.min())
    if A_max - A_min == 0:
        # should be descriptive later
        raise
        return A
    else:
        return 255 * (A - A_min) / (A_max - A_min)

# this is for interactive debugging. fix your code!
show = lambda x: Image.fromarray(x).show()

if __name__ == "__main__":

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

    # NOW THIS IS THE A IN QUESTION
    A = np.array(vectors, dtype='f').T


    rank_one = rank_one_approx(A)
    fg = A - rank_one

    # too many steps going on here :/
    fg_matrices = [normalize_grayscale(column).reshape((512,512)) for column in fg.T]
    bg_matrices = [normalize_grayscale(column).reshape((512,512)) for column in rank_one.T]


    # apparently the tiff plugin fails if the pixels are still floats
    for i, image in enumerate(fg_matrices):
        im = Image.fromarray(image.astype('uint8'))
        filestring = 'data/MOTION/fg_{}.tiff'.format(i+1)
        im.save(filestring)

    for i, image in enumerate(bg_matrices):
        im = Image.fromarray(image.astype('uint8'))
        filestring = 'data/MOTION/bg_{}.tiff'.format(i+1)
        im.save(filestring)


