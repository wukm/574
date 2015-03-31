#!/usr/bin/env python3

"""
loadFACES.py

loads the 'Extended YaleB' facial recognition data set

there are 2414 images total of 38 subjects (~64 images each)

each image is a grayscale 192x168
"""
import os
import os.path
import PIL.Image
import numpy


YALEPATH = './data/YaleB/'

def img_file_to_row_vector(filename):
    """
    look at this function first if the data is garbled
    not sure what dtype should be or if data stays in the right order
    """
    img = PIL.Image.open(filename)
    img_m = numpy.array(img) # will be a 2d array dtype=uint8
    # make it a row vector
    return img_m.reshape((1,-1))


def make_all_faces(path=None):
    """
    this is really fast, like < .2 sec
    even though it schleps a lot of data about. so no pickle
    """
    
    if path is None:
        path = YALEPATH

    X = None
    lab = None

    # get all the subdirectories. each element of list will be
    # (dirname, dircontents)
    subdirs = [(dtup[0], dtup[2]) for dtup in os.walk(YALEPATH) if not dtup[1]]

    # put these in numerical order, so we get classes in numerical order, just for
    # consistency
    subdirs.sort(key=lambda d: d[0])

    for dirname, contents in subdirs:
        # this is redundant since they're sorted, but hey
        class_no = int(dirname[-2:]) # yaleB##
        
        row_vectors = [img_file_to_row_vector(os.path.join(dirname, filepath)) for
                filepath in contents]
    
        A = numpy.vstack(row_vectors)
        L = numpy.array([class_no]*len(contents))
        L = L.reshape((-1,1))

        if X is None:
            X = A
            lab = L
        else:
            X = numpy.vstack((X,A))
            lab = numpy.vstack((lab,L))

    return X, lab

def class_filter(classes, X, b):
    """
    if classes is nonempty, do class filtering
    """

    # keeping the filtering in numpy would be faster i bet
    relevant = [(X[i], y) for i,y in enumerate(b) if y in classes]

    X = numpy.array([x[0] for x in relevant])
    b = numpy.array([x[1] for x in relevant])
    
    return X, b

def load_faces(classes=None, typestr='all', path=None):
    """
    loads the extended YaleB facial recognition dataset
    typestr must be one of 'all', 'train', or 'test'
    if not specified, it defaults to 'all'

    in this case 'train' will be a 'random' subset of the entire dataset, and
    'train' will be in complement.
    
    returns a matrix faces" whose rows are the vectorized images in question
    and "labels", which is a 2D column vector of corresponding class labels 1-39
    """

    if path is None:
        path = YALEPATH

    if typestr not in ('all', 'train', 'test'):
        raise Exception("need to specify 'all', 'train', or 'test'")

    else:
        X, lab = make_all_faces(path=path)
    
        it = numpy.loadtxt(os.path.join(path, 'train_indices.txt'),
        dtype='uint8')

        if typestr == 'train':
            faces = X[it -1]
            labels = lab[it -1]

        elif typestr == 'test':
            nit = numpy.array([i for i in range(1,X.shape[0]+1) if i not in it])
            faces = X[nit -1]
            labels = lab[nit - 1]
        
        else:
            faces, labels = X, lab

    # do class filtering if 'classes' was a nonempty iterable
    if classes:
        faces, labels = class_filter(classes, faces, labels)

    # finally, transform faces from uint8 to doubles in [0,1]: [0,255] -> [0,1]
    return faces/255., labels

if __name__ == "__main__":

    X, lab = make_all_faces()
