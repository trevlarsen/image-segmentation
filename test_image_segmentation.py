"""Unit testing file for Image Segmentation"""

import image_segmenter
import pytest
import numpy as np
from scipy.sparse import load_npz

def test_adjacency():
    yuh = image_segmenter.ImageSegmenter('image_files/blue_heart.png')
    A, D = yuh.adjacency()

    # Load the saved adjacency matrix as CSC format
    A2 = load_npz('image_files/HeartMatrixA.npz')
    
    # Ensure both matrices are in CSC format
    A = A.tocsc()
    A2 = A2.tocsc()

    # Compare components separately to allow tolerance on data
    assert A.shape == A2.shape, "Shape mismatch between A and A2"
    assert np.array_equal(A.indices, A2.indices), "Indices mismatch in adjacency matrix"
    assert np.array_equal(A.indptr, A2.indptr), "Indptr mismatch in adjacency matrix"
    assert np.allclose(A.data, A2.data, atol=1e-5), "Data values mismatch in adjacency matrix"

    # Load and compare degree matrices
    D2 = np.load('image_files/HeartMatrixD.npy')
    assert np.allclose(D, D2), 'Degree matrix comparison failed'
    
    print("All tests passed.")

