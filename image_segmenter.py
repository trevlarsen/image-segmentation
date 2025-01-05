import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
from imageio.v2 import imread

class ImageSegmenter:
    """Class for loading, visualizing, and segmenting images based on brightness and spatial proximity.

    Attributes:
        scaled (ndarray): Scaled image data in the range [0, 1].
        brightness (ndarray): Flattened array of pixel brightness values.
        m (int): Height of the image.
        n (int): Width of the image.
    """

    def __init__(self, filename):
        """Initialize the ImageSegmenter with image data.

        Args:
            filename (str): Path to the image file.
        """
        image = imread(filename)
        self.scaled = image / 255
        self.m, self.n = self.scaled.shape[:2]

        if self.scaled.ndim == 3:
            self.brightness = self.scaled.mean(axis=2).ravel()
        else:
            self.brightness = self.scaled.ravel()

    def show_original(self):
        """Display the original image."""
        cmap = None if self.scaled.ndim == 3 else 'gray'
        plt.imshow(self.scaled, cmap=cmap)
        plt.axis('off')
        plt.show()

    def adjacency(self, r=5.0, sigma_B2=0.02, sigma_X2=3.0):
        """Compute the adjacency and degree matrices for the image graph.

        Args:
            r (float): Radius for neighbor inclusion.
            sigma_B2 (float): Variance for brightness similarity.
            sigma_X2 (float): Variance for spatial proximity.

        Returns:
            sp.csc_matrix: Adjacency matrix.
            ndarray: Degree matrix diagonal elements.
        """
        mn = self.brightness.size
        A = sp.lil_matrix((mn, mn))
        D = np.zeros(mn)

        for i in range(mn):
            neighbors, distances = self._get_neighbors(i, r)
            weights = np.exp(-((np.abs(self.brightness[i] - self.brightness[neighbors]) / sigma_B2) + (distances / sigma_X2)))
            A[i, neighbors] = weights
            D[i] = weights.sum()

        return A.tocsc(), D

    def _get_neighbors(self, index, radius):
        """Find neighboring pixels within a radius and compute distances.

        Args:
            index (int): Index of the central pixel in the flattened array.
            radius (float): Radius for neighbor inclusion.

        Returns:
            ndarray: Indices of neighbors.
            ndarray: Distances to neighbors.
        """
        row, col = divmod(index, self.n)
        r = int(radius)
        x = np.arange(max(col - r, 0), min(col + r + 1, self.n))
        y = np.arange(max(row - r, 0), min(row + r + 1, self.m))
        X, Y = np.meshgrid(x, y)
        R = np.sqrt((X - col)**2 + (Y - row)**2)
        mask = R < radius
        return (X[mask] + Y[mask] * self.n).astype(int), R[mask]

    def cut(self, A, D):
        """Compute the boolean mask that segments the image.

        Args:
            A (sp.csc_matrix): Adjacency matrix.
            D (ndarray): Degree matrix diagonal elements.

        Returns:
            ndarray: Boolean mask for segmentation.
        """
        L = sp.csgraph.laplacian(A)
        D_inv_sqrt = sp.diags(D**-0.5)
        normalized_L = D_inv_sqrt @ L @ D_inv_sqrt
        _, eigvecs = sla.eigsh(normalized_L, which='SM', k=2)
        second_smallest = eigvecs[:, 1]
        return second_smallest.reshape((self.m, self.n)) > 0

    def segment(self, r=5.0, sigma_B=0.02, sigma_X=3.0, save_image=False):
        """Segment the image and display the results.

        Args:
            r (float): Radius for neighbor inclusion.
            sigma_B (float): Variance for brightness similarity.
            sigma_X (float): Variance for spatial proximity.
        """
        A, D = self.adjacency(r, sigma_B, sigma_X)
        mask = self.cut(A, D)

        if self.scaled.ndim == 3:
            mask = np.dstack((mask, mask, mask))

        pos = self.scaled * mask
        neg = self.scaled * ~mask

        plt.figure(figsize=(12, 4), dpi=300)
        plt.subplot(131)
        plt.title("Original")
        plt.imshow(self.scaled, cmap=None if self.scaled.ndim == 3 else 'gray')
        plt.axis('off')
        plt.subplot(132)
        plt.title("Segment 1")
        plt.imshow(pos, cmap=None if self.scaled.ndim == 3 else 'gray')
        plt.axis('off')
        plt.subplot(133)
        plt.title("Segment 2")
        plt.imshow(neg, cmap=None if self.scaled.ndim == 3 else 'gray')
        plt.axis('off')
        if save_image: plt.savefig('demo.png')
        plt.show()