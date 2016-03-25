"""
Linear algebra utilities.
"""

import numpy as np

from functools import reduce
from math import acos, atan2


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
    'normalize_rows',
    'to_unit_vector',
    'dagger',
    'kron',
    'cartesian_to_spherical',
    'random_direction_angles',
    'random_direction_vector',
    'random_quantum_state',
    'to_quantum_state',
    'complex2real',
    'expectation_value',
    'measurement',
    'as_column_vector',
    'projector',
    'solve_quadratic_equation',
    'hdet',
    'ptrace',
    'ptranspose',
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

    The basis vectors are returned as row vectors.
    """
    v = np.atleast_2d(v)
    a = np.hstack((v.T , np.eye(v.shape[1])))
    q, r = np.linalg.qr(a)
    return q[:,1:].T


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
    """Normalize all rows in a matrix."""
    return M / np.linalg.norm(M, axis=-1)[:,np.newaxis]


def to_unit_vector(v):
    """Normalize a cartesian vector."""
    return v / np.linalg.norm(v)


def dagger(M):
    """Return the transpose conjugate (=adjoint) of a complex matrix."""
    return M.conj().T


def kron(*parts):
    """Compute the repeated Kronecker product (tensor product)."""
    return reduce(np.kron, parts)


def cartesian_to_spherical(v):
    """
    Coordinate transformation from Cartesian to spherical.

    For an input vector of the form

        r sin(θ) cos(φ)
        r sin(θ) sin(φ)
        r cos(θ)

    Returns (r, theta, phi).
    """
    r = np.linalg.norm(v)
    theta = acos(v[2]/r)
    phi = atan2(v[1], v[0])
    return (r, theta, phi)


def hyperspherical_to_cartesian(angles):
    """
    Convert hyperspherical angles to cartesian coordinates.

    Converts a vector of n angles to a (n+1) cartesian unit vector with
    components

        x₁   = cos(φ₁)
        x₂   = sin(φ₁) cos(φ₂)
        x₃   = sin(φ₁) sin(φ₂) cos(φ₃)
        …
        xₙ   = sin(φ₁) sin(φ₂) sin(φ₃) … sin(φₙ₋₁) cos(φₙ)
        xₙ₊₁ = sin(φ₁) sin(φ₂) sin(φ₃) … sin(φₙ₋₁) sin(φₙ)
    """
    dim = angles.size + 1
    c = [cos(x) for x in angles] + [1]
    s = [1] + [sin(x) for x in angles]
    return np.array([ci * si for ci, si in zip(c, np.cumprod(s))])


def random_direction_angles():
    """Return unit vector on the sphere in spherical coordinates (θ, φ)."""
    v = np.random.normal(size=3)
    r, theta, phi = cartesian_to_spherical(v)
    return theta, phi


def random_direction_vector(size):
    """Return unit vector on the sphere in cartesian coordinates (x, y, z)."""
    return to_unit_vector(np.random.normal(size=size))


def random_quantum_state(dim):
    """Return a random quantum state."""
    return to_quantum_state(random_direction_vector((dim, 2)))


def to_quantum_state(coefs):
    """Convert a sequence of coefficient tuples to a complex state vector."""
    return np.array([complex(a, b) for a, b in coefs])


def complex2real(z: complex, eps=1e-13) -> float:
    """
    Convert a complex number to a real number.

    Use this function after calculating the expectation value of a hermitian
    operator.
    """
    if z.imag > eps:
        raise ValueError("{} is not a real number.".format(z))
    return z.real


def expectation_value(psi, M) -> complex:
    """
    Return the expectation value <ψ|M|ψ>.

    :param np.ndarray psi: vector m*1
    :param np.ndarray M: matrix m*m
    """
    return dagger(psi) @ M @ psi


def measurement(psi, M) -> float:
    """Return the measurement <ψ|M|ψ> of a hermitian operator."""
    return complex2real(expectation_value(psi, M))


def as_column_vector(vec):
    """Reshape the array to a column vector."""
    vec = np.asarray(vec)
    return vec.reshape((vec.size, 1))


def projector(vec):
    """
    Return the projection matrix to the 1D space spanned by the given vector.
    """
    vec = as_column_vector(vec)
    return vec @ dagger(vec)


def solve_quadratic_equation(a, b, c):
    """
    Return the two solutions (x₊, x₋) of the quadratic equation
    ``ax² + bx + c = 0``.
    """
    radix = cmath.sqrt(b**2 - 4*a*c)
    return ((-b + radix) / (2*a),
            (-b - radix) / (2*a))


def hdet(a):
    """
    Compute the hyperdeterminant of a 2x2x2 matrix (Cayley's Hyperdeterminant).

    https://en.wikipedia.org/wiki/Hyperdeterminant
    """
    return (
        + a[0,0,0]**2 * a[1,1,1]**2
        + a[0,0,1]**2 * a[1,1,0]**2
        + a[0,1,0]**2 * a[1,0,1]**2
        + a[1,0,0]**2 * a[0,1,1]**2
        - 2 * a[0,0,0] * a[0,0,1] * a[1,1,0] * a[1,1,1]
        - 2 * a[0,0,0] * a[0,1,0] * a[1,0,1] * a[1,1,1]
        - 2 * a[0,0,0] * a[1,0,0] * a[0,1,1] * a[1,1,1]
        - 2 * a[0,0,1] * a[0,1,0] * a[1,0,1] * a[1,1,0]
        - 2 * a[0,0,1] * a[0,1,1] * a[1,1,0] * a[1,0,0]
        - 2 * a[0,1,0] * a[0,1,1] * a[1,0,1] * a[1,0,0]
        + 4 * a[0,0,0] * a[0,1,1] * a[1,0,1] * a[1,1,0]
        + 4 * a[0,0,1] * a[0,1,0] * a[1,0,0] * a[1,1,1]
    )


def hdet_alt(a):
    """
    This is alternative formula for the hyperdeterminant of a 2x2x2 matrix
    taken from https://en.wikipedia.org/wiki/Hyperdeterminant and implemented
    to provide a comparison for the `hdet` function. Unfortunately, something
    seems to be wrong with the formula (or implementation). I'm still keeping
    it here to investigate the issue at some point.
    """
    eps = np.array([[0, 1], [-1, 0]])
    b = np.einsum('il,jm,ijk,lmn', eps, eps, a, a) / 2
    d = np.einsum('il,jm,ij,lm', eps, eps, b, b) / 2
    return d


def ptrace(dm, dims, *out):
    """
    Perform partial trace on a density matrix. Trace out all subsystems with
    the specified indices.
    """
    dims = list(dims)
    dm = dm.reshape(dims * 2)
    for rm in sorted(out, reverse=True):
        dm = dm.trace(axis1=rm,
                      axis2=rm+len(dims))
        del dims[rm]
    dim = np.product(dims)
    dm = dm.reshape((dim, dim))
    return dm


def ptranspose(dm, dims, subsys):
    """
    Return the partial transpose of a density matrix with respect to the
    given subsystem.
    """
    n = len(dims)
    d = np.product(dims)
    dims = list(dims)
    perm = list(range(n*2))
    perm[subsys], perm[n+subsys] = n+subsys, subsys
    return dm.reshape(dims * 2).transpose(tuple(perm)).reshape((d, d))
