"""
Linear algebra utilities.
"""

import numpy as np


__all__ = [
    'matrix_rank',
    'matrix_rowspace',
    'matrix_nullspace',
    'matrix_imker',
    'matrix_imker_nice',
    'plane_basis',
    'plane_normal',
    'basis_vector',
    'project_to_plane',
]


matrix_rank = np.linalg.matrix_rank


def matrix_rowspace(A, eps=1e-10):
    """Return matrix row space, i.e. Im[A.T]."""
    return matrix_im_ker(A, eps)[1]


def matrix_nullspace(A, eps=1e-10):
    """Return a basis for the solution space of ``A∙x=0`` as row vectors."""
    return matrix_imker(A, eps)[1]


def matrix_imker(A, eps=1e-10):
    """
    Return bases for ``(Im[A.T], Ker[A])`` as row vectors.
    """
    u, s, vh = np.linalg.svd(A)
    n = next((i for i, c in enumerate(s) if c < eps), len(s))
    return vh[:n], vh[n:]


def matrix_imker_nice(A, eps=1e-10):
    """Get (Im[A], Ker[A]) but try to return pretty matrices."""
    m, n = A.shape
    img, ker = matrix_imker(A, eps)
    eye = np.eye(n)
    nil = np.empty((0, n))
    # Short-cut for the case where either image or kernel is the full space
    if len(img) == n: return eye, nil
    if len(ker) == n: return nil, eye
    # Find Euclidean subspaces of image and kernel (=good indices)
    allspace = range(n)
    img_good = [i for i in allspace
                if all(abs(v) < eps for v in ker[:,i])]
    ker_good = [i for i in sorted(set(allspace) - set(img_good))
                if all(abs(v) < eps for v in img[:,i])]
    any_good = sorted(img_good + ker_good)
    all_ugly = sorted(set(allspace) - set(any_good))
    # Another shortcut for when both image and kernel are Euclidean
    if len(all_ugly) == n: return img, ker
    if len(any_good) == n: return eye[img_good], eye[ker_good]
    # Determine ONB in the non-Euclidean subspaces
    img, ker = matrix_imker(A[:,all_ugly])
    insertion_indices = [v-i for i, v in enumerate(any_good)]
    img = np.insert(img, insertion_indices, 0, axis=1)
    ker = np.insert(ker, insertion_indices, 0, axis=1)
    img = np.vstack((eye[img_good], img))
    ker = np.vstack((eye[ker_good], ker))
    return img, ker


def basis_vector(dim, index):
    """Get D dimensional unit vector with only the i-th component being 1."""
    v = np.zeros(dim)
    v[index] = 1
    return v


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


def project_to_plane(v, n):
    """Project v into the subspace defined by x∙n = 0."""
    return v - n * (v @ n) / np.linalg.norm(n)


def normalize_rows(M):
    return M / np.linalg.norm(M, axis=-1)[:,np.newaxis]


def addz(mat):
    mat = np.atleast_2d(mat)
    return np.hstack((np.zeros((mat.shape[0], 1)), mat))


def delz(mat):
    mat = np.atleast_2d(mat)
    return mat[:,1:]
