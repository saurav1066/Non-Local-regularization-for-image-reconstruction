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

from functions import *


class NLRAlgorithm:
    """
    Main class that implements the regularization algorithm of the paper

    For its initialization it requires:

    The matrix Phi
    The measurement vector, u
    The size of the squares used for the graph (tau un the paper)
    The radius used to define neighbour pixels, rho
    """

    def __init__(self, _Phi, _u, _square_size, _q, _proj_image, _rho, _lambda, _maxIter, _use_patches):
        self.Phi = _Phi
        self.Phi_diagonal = self.Phi.diagonal()
        self.Phi_star = self.Phi.transpose()
        self.N = np.shape(_Phi)[0]
        self.n = int(math.sqrt(self.N))
        self.u = _u
        self.square_size = _square_size
        self.rho = _rho
        self.q = _q
        self.max_iter = _maxIter
        self.w = None
        self.gamma = np.max(self.u) 
        self.use_patches = _use_patches

        self.lmbda = _lambda

        print("Image size = ", self.n, " x ", self.n)

        # f = Phi* u
        self.f_vec = self.u
        self.f_mat = vector_to_matrix(self.f_vec)

        # Matrix of size N x 2 that contains the x and y coordinates for each pixel
        self.xy_coords = np.column_stack(np.where(self.f_mat >= -1))

        self.orth_projector = self.compute_orth_projector(_proj_image)
        self.orth_proj_star = self.orth_projector.transpose()
        self.patches = None
        
    def plot_image(self):
        plt.figure(figsize=(5, 5))
        imageplot(vector_to_matrix(self.f_vec))
        plt.show()
        return None
        
        
    def find_neighbours(self, pixelID, rho):
        # We choose the neighours to be all the pixels inside
        # a square of side rho
        x, y = self.get_pixel_coord(pixelID)

        neighbours_list = []
        for i in range(max(0, x - rho), min(self.n, x + rho + 1)):
            for j in range(max(0, y - rho), min(self.n, y + rho + 1)):
                neighbours_list.append(self.get_pixel_ID_from_coord(i, j))
#         neighbours_list = [i for i in neighbours_list if self.Phi_diagonal[i] != 0]
        neighbours_list = np.array(neighbours_list)
        return neighbours_list[(neighbours_list != pixelID)]
            

    def pixels_distance(self, id1, id2):
        # Returns the distance between two given pixels
        x1, y1 = self.get_pixel_coord(id1)
        x2, y2 = self.get_pixel_coord(id2)
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_pixel_coord(self, pixelID):
        # I assume the origin to be at the upper left corner of the image
        return [self.xy_coords[pixelID][0], self.xy_coords[pixelID][1]]

    def get_pixel_ID_from_coord(self, _x, _y):
        # I assume the origin to be at the upper left corner of the image
        return int(_x * self.n + _y)


    def get_patches(self):
        pad_width = self.square_size // 2
        image = pad(self.f_vec.reshape((self.n, self.n)), pad_width, mode="symmetric")

        patches = extract_patches_2d(image, (self.square_size, self.square_size))

        # Reshape patches into a 3D array
        patches = patches.reshape((-1, self.square_size, self.square_size))
        
        num_patches, patch_size, _ = patches.shape
        patches_2d = patches.reshape((num_patches, patch_size**2))

        # Project patches
        projected_patches = patches_2d.dot(self.orth_projector.T)

        # Reshape projected patches into a 3D array
        projected_patches = projected_patches.reshape((patches.shape[0], self.q))
        
        return projected_patches


    def compute_orth_projector(self, image):

        patch_list = []
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        patches = extract_patches_2d(image, (self.square_size, self.square_size), max_patches=None)
        patch_list.extend(patches)

        # Convert patches to flattened vectors
        patch_vectors = np.array([patch.flatten() for patch in patch_list])

        # Compute the PCA eigenvectors of the patches
        pca = PCA(n_components=self.q)
        pca.fit(patch_vectors)
        U = pca.components_
        return U


    def weight_update(self, _rho, use_patches):
        print("\nStart weight update")
        
        gamma = self.square_size * np.max(self.f_vec) / 10
        weights = scp.sparse.lil_matrix((self.N, self.N))
        
        if use_patches:
            for i in range(self.N):
                x = self.patches[i]
                neighbor_indices = self.find_neighbours(i, _rho)
                d = np.linalg.norm(x - self.patches[neighbor_indices], axis=1)
                w = np.exp(-d / gamma)
                Z = np.sum(w)
                weights[i, neighbor_indices] = w / Z
        else:
            for i in range(self.N):
                x = self.f_vec[i]
                neighbor_indices = self.find_neighbours(i, _rho)
                d = np.abs(x - self.f_vec[neighbor_indices])
                w = np.exp(-d / gamma)
                Z = np.sum(w)
                weights[i, neighbor_indices] = w / Z
            
        # Convert the weights to CSR format for faster matrix multiplication
        self.w = weights.tocsr()
        self.r = self.w
        return None
    

    ## TODO
    def image_update(self):
        print("image update")
        row_sums = np.sum(self.w, axis=1)
        print(row_sums.shape)
        self.w = -self.w
        self.w.setdiag(row_sums)
        
        self.f_vec = scp.sparse.linalg.gmres(self.Phi.T @ self.Phi + self.lmbda * self.w, 
                                             self.Phi.T.dot(self.u))[0]

        return None

    
    def main_loop(self):
        for i in range(self.max_iter):
            
            print(f"Iteration {i}".center(10))
            self.patches = self.get_patches()
            self.weight_update(self.rho, self.use_patches)
            self.image_update()
            end_time = time.time()

            self.plot_image()
            
        return vector_to_matrix(self.f_vec)