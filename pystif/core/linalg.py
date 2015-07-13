"""
Linear algebra utilities.
"""

import numpy as np


__all__ = [
    'matrix_rank',
    'matrix_rowspace',
    'matrix_nullspace',
    'matrix_imker',
    'plane_basis',
    'plane_normal',
    'PCA',
    'basis_vector',
    'project_to_plane',
]


matrix_rank = np.linalg.matrix_rank


def matrix_rowspace(A, eps=1e-10):
    """Return matrix row space, i.e. Im[A]."""
    u, s, vh = np.linalg.svd(A)
    n = next((i for i, c in enumerate(s) if c < eps), len(s))
    return vh[:n]


def matrix_nullspace(A, eps=1e-10):
    """
    Return a basis for the solution space of ``A∙x=0`` as row vectors.
    """
    u, s, vh = np.linalg.svd(A)
    n = next((i for i, c in enumerate(s) if c < eps), len(s))
    return vh[n:]


def matrix_imker(A, eps=1e-10):
    """
    Return bases for ``(Im[A], Ker[A])`` as row vectors.
    """
    u, s, vh = np.linalg.svd(A)
    n = next((i for i, c in enumerate(s) if c < eps), len(s))
    return vh[:n], vh[n:]


def plane_basis(v):
    """
    Get the (orthonormal) basis vectors making up the orthogonal complement of
    the plane defined by v∙x = 0.
    """
    v = np.atleast_2d(v)
    a = np.hstack((v.T , np.eye(v.shape[1])))
    q, r = np.linalg.qr(a)
    return q[:,1:]


def plane_normal(v):
    """
    Get normal vector of the plane defined by the vertices (=rows of) v.
    """
    space = matrix_nullspace(v)
    assert space.shape[0] == 1
    return space[0].flatten()


def PCA(data_points, eps=1e-10):
    """
    Get the (orthonormal) basis vectors of the principal components of the
    data set specified by the rows of M.
    """
    return matrix_imker(np.cov(data_points.T), eps)


def project_to_plane(v, n):
    """Project v into the subspace defined by x∙n = 0."""
    return v - n * np.dot(v, n) / np.linalg.norm(n)


def normalize_rows(M):
    return M / np.linalg.norm(M, axis=-1)[:,np.newaxis]


def addz(mat):
    mat = np.atleast_2d(mat)
    return np.hstack((np.zeros((mat.shape[0], 1)), mat))


def delz(mat):
    mat = np.atleast_2d(mat)
    return mat[:,1:]
