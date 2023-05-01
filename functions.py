import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
import math
from scipy.sparse import csr_matrix
import cv2
import pylab
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import PCA
from PIL import Image
from numpy.lib.arraypad import pad
from numpy import random
import time

def createPhi(n, percentage):
    np.random.seed(4444)
    indices = random.choice(n**2, size=int(n**2*percentage), replace=False, p=None)
    matrix = scp.sparse.lil_matrix((n**2, n**2))
    for i in indices:
        matrix[i, i] = 1
    return csr_matrix(matrix)

def imageplot(f, str='', sbpt=[]):
    """
        Use nearest neighbor interpolation for the display.
    """
    if sbpt != []:
        plt.subplot(sbpt[0], sbpt[1], sbpt[2])
    imgplot = plt.imshow(f, interpolation='nearest')
    imgplot.set_cmap('gray')
    pylab.axis('off')
    if str != '':
        plt.title(str)

def matrix_to_vector(image):
    # Convert image from matrix to vector
    return image.flatten()


def vector_to_matrix(image):
    # Convert image from vector to matrix
    return image.reshape(int(math.sqrt(image.size)), -1)

def psnr(x, y, vmax=-1):
    """
     psnr - compute the Peack Signal to Noise Ratio

       p = psnr(x,y,vmax);

       defined by :
           p = 10*log10( vmax^2 / |x-y|^2 )
       |x-y|^2 = mean( (x(:)-y(:)).^2 )
       if vmax is ommited, then
           vmax = max(max(x(:)),max(y(:)))

       Copyright (c) 2014 Gabriel Peyre
    """

    if vmax < 0:
        m1 = abs(x).max()
        m2 = abs(y).max()
        vmax = max(m1, m2)
    d = np.mean((x - y) ** 2)
    return 10 * np.log10(vmax ** 2 / d)

def load_image(name, n=-1, flatten=1, resc=1, grayscale=1):
    """
        Load an image from a file, rescale its dynamic to [0,1], turn it into a grayscale image
        and resize it to size n x n.
    """
    f = plt.imread(name)
    # turn into normalized grayscale image
    if grayscale == 1:
        if (flatten==1) and np.ndim(f)>2:
            f = np.sum(f, axis=2)
    if resc==1:
        f = rescale(f)
    # change the size of the image
    if n > 0:
        if np.ndim(f)==2:
            f = transform.resize(f, [n, n], 1)
        elif np.ndim(f)==3:
            f = transform.resize(f, [n, n, f.shape[2]], 1)
    return f

def rescale(f,a=0,b=1):
    """
        Rescale linearly the dynamic of a vector to fit within a range [a,b]
    """
    v = f.max() - f.min()
    g = (f - f.min()).copy()
    if v > 0:
        g = g / v
    return a + g*(b-a)