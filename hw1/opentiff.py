#!/usr/bin/python3

# this shit opens tiff files. trying to figure out how this works, get it into a
# matrix of pixels.

from PIL import Image
import numpy as np
from scipy import linalg as sla 
# hope you're in the right directory you sad shit ;)
image_files = [''.join(('motion_',str(n),'.tiff')) for n in range(1,11)]


# note that np.array() can accept either im by itself (which it converts to a
# matrix), or im.getdata() to immediately columnize it. prolly a bunch of other
# ways too.
vectors = []

# this is a little slow. rewrite with generators?
for image in image_files:
    im = Image.open(image)
    vectors.append(np.array(im.getdata()))

A = np.array(vectors)

