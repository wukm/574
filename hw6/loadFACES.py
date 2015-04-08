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

def img_file_to_row_vector(filename, dtype=None):
    """
    image file to numpy array with the same 
    look at this function first if the data is garbled
    not sure what dtype should be or if data stays in the right order
    """
    img = PIL.Image.open(filename)
    img_m = numpy.array(img, dtype='double') # will be a 2d array dtype=uint8

    # make it a row vector
    img_m = img_m.reshape((1,-1))

    if dtype is not None:
        img_m = img_m.astype(dtype)

    return img_m

def make_all_faces(path=None):
    """
    this is really fast, like < .2 sec
    even though it schleps a lot of data about. so no pickle
    """
    
    if path is None:
        path = YALEPATH

    # this is for the loop before so I don't have to initialize these with a
    # particular size. maybe change this
    X, lab = None, None

    # get all the subdirectories. each element of list will be
    # (dirname, dircontents)
    # the condition at the end is to make sure that we're only returning
    # directories with no further subdirectories (this should match every
    # directory in question except YALEPATH itself)
    subdirs = [(dtup[0], dtup[2]) for dtup in os.walk(YALEPATH) if not dtup[1]]

    # put these in numerical order, so we get classes in numerical order, just for
    # consistency
    subdirs.sort(key=lambda d: d[0])

    for dirname, contents in subdirs:
        # because the directories look like yaleB(##)
        class_no = int(dirname[-2:])
        
        row_vectors = [img_file_to_row_vector(os.path.join(dirname, filepath)) for
                filepath in contents]
    
        # make matrix with the image in this subdrirectory as rows
        A = numpy.vstack(row_vectors)
        # make a (-1,1) shaped array with the class label for these images
        L = numpy.array([class_no]*len(contents), dtype='int')
        L = L.reshape((-1,1))

        if X is None:
            X = A
            lab = L
        else:
            X = numpy.vstack((X,A))
            lab = numpy.vstack((lab,L))

    return X, lab

def class_filter2(classes, X, labels):
    """
    implementation with a listcomp. stupid and deprecated.
    filter by the classes passed through to load_faces
    """
    
    
    relevant = [(X[i], label) for i, labels in enumerate(labels) if y in classes]

    X = numpy.array([x[0] for x in relevant])
    b = numpy.array([x[1] for x in relevant])
    
    return X, b

def class_filter(classes, X, labels):
    """
    hey, kept in numpy now!
    this removes rows from X and labels that don't
    correspond to anything in classes
    """
    classes = numpy.array(classes)
    supp = numpy.nonzero(classes == labels)[0]
    
    A = X[supp]
    b = labels[supp] 

    return A, b

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
        X, lab = make_all_faces(path)
    
        # 'it' will return a 1d array. beware that the indices here actually
        # refer to matlab indices, so need to subtract 1 for everything
        # also note that the indices in train_indices.txt are not in order for
        # whatever reason. not that it matters
        it = numpy.loadtxt(os.path.join(path, 'train_indices.txt'),
        dtype='int')

        it -= 1 # matlab is 1-indexed, python is 0-indexed

        if typestr == 'train':
            faces = X[it]
            labels = lab[it]
        
        # it these should be X\X[it] and lab\lab[it]
        elif typestr == 'test':
            #nit = numpy.array([i for i in range(X.shape[0]) if i not in it])
            nit = numpy.setdiff1d(numpy.arange(X.shape[0]), it) 
            faces = X[nit]
            labels = lab[nit]
        
        else:
            faces, labels = X, lab

    # do class filtering if 'classes' was nonempty
    if classes:
        faces, labels = class_filter(classes, faces, labels)

    # finally, transform faces from ints to doubles in [0,1]: [0,255] -> [0,1]
    faces = faces / 255.

    return faces, labels.astype('d')

if __name__ == "__main__":

    X, lab = load_faces(list(range(1,38)), 'train')
