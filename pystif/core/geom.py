
import numpy as np

from .array import scale_to_int, make_int_exact
from .linalg import (matrix_imker, matrix_nullspace,
                     basis_vector, plane_normal, addz, delz)
from .util import PointSet, cached


def random_direction_vector(dim):
    v = np.random.normal(size=dim)
    v /= np.linalg.norm(v)
    return v


class ConvexPolyhedron:

    """
    Utility for finding the extreme points (vertices) of a convex polyhedron
    defined by the projection of a system of linear inequalities.

    Assumes that the origin is an extreme point of the polyhedron.
    """

    def __init__(self, lp, dim):
        self.lp = lp
        self.dim = dim
        self.points = PointSet()

    @classmethod
    def from_cone(cls, system, dim, limit):
        """Create search problem from the system ``L∙x≥0, x≤limit``."""
        lp = system.lp()
        for i in range(1, dim):
            lp.set_col_bnds(i, 0, limit)
        return cls(lp, dim)

    @cached
    def basis(self):
        """
        Return a matrix of extreme points whose convex hull defines an inner
        approximation to the projection of the polytope defined by the LP to
        the subspace of its lowest ``self.dim`` components.

        Extreme points are returned as matrix rows.

        The returned points define a polytope with the dimension of the
        projected polytope itself (which may be less than ``self.dim``).
        """
        points = np.empty((0, self.dim))
        orth = LinearSubspace.all_space(self.dim-1)
        while orth.dim > 0:
            # Choose vector from orthogonal space and optimize along its
            # direction:
            d = random_direction_vector(orth.dim)
            v = orth.back(d)
            v = np.hstack((0, v))
            x = self.search(v)[1:]
            p = make_int_exact(orth.into(x))
            if all(p == 0):
                x = self.search(-v)[1:]
                p = make_int_exact(orth.into(x))
            if all(p == 0):
                # Optimizing along ``v`` yields a vector in our ray space.
                # This means ``v∙x=0`` is part of the LP.
                orth = orth.back_space(LinearSubspace.from_nullspace(d))
            else:
                # Remove discovered ray from the orthogonal space:
                orth = orth.back_space(LinearSubspace.from_nullspace(p))
                points = np.vstack((
                    points,
                    np.hstack((0, x)),
                ))
        return points

    @cached
    def subspace(self):
        return LinearSubspace.from_rowspace(delz(self.basis()))

    def search(self, q):
        """
        Search an extreme point ``x`` of the LP which minimizes ``q∙x``.

        The ``q`` parameter must be specified as a vector with ``dim``
        components. Its geometric interpretation is the normal vector of a
        hyperplane in the projection space. This hyperplane is shifted along
        its normal until all points inside the polyhedron fulfill ``q∙x≥0``.

        Returns an extreme point with ``dim`` components (the leftmost
        component is always 0).
        """
        assert len(q) == self.dim
        extreme_point = self.lp.minimize(q, embed=True)
        extreme_point = extreme_point[0:self.dim]
        extreme_point[0] = 0
        extreme_point = scale_to_int(extreme_point)
        self.points.add(extreme_point)
        return extreme_point

    def intersection(self, space):
        """
        Return the ConvexPolyhedron obtained from the intersection with the
        given subspace.

        :param space: specified by its normal vector(s).
        """
        space = np.atleast_2d(space)
        assert space.shape[1] == self.dim
        lp = self.lp.copy()
        lp.add(space, 0, 0, embed=True)
        return self.__class__(lp, self.dim)

    @cached
    def rank(self):
        """Geometric dimension of the projected polyhedron."""
        return len(self.basis())

    def face_rank(self, face):
        """Return dimension of the face."""
        face = self.subspace().projection(face)
        return self.intersection(face).rank()

    def is_facet(self, face):
        return self.is_face(face) and self.face_rank(face) == self.rank()-1

    def is_face(self, face):
        return self.lp.implies(face, embed=True)

    def get_adjacent_facet(self, face, inner, atol=1e-10):
        """
        Get the adjacent facet defined by `facet` and `subface`.
        """
        plane = -face
        while True:
            vertex = self.search(plane)
            if self.lp.get_objective_value() >= -atol:
                # assert self.is_face(plane)
                # assert self.face_rank(plane) >= self.face_rank(face)
                return scale_to_int(plane), scale_to_int(inner)
            plane /= np.linalg.norm(plane)
            inner /= np.linalg.norm(inner)
            fx = plane @ vertex
            sx = inner @ vertex
            plane, inner = (
                inner - sx/fx * plane,
                fx * plane + sx * inner,
            )

    def filter_non_singular_directions(self, nullspace):
        while nullspace.shape[0] > 0:
            direction = random_direction_vector(nullspace.shape[0])
            direction = nullspace.T @ direction
            if self.is_face(direction):
                direction = -direction
                if self.is_face(direction):
                    # TODO: remove direction from nullspace
                    return
            yield direction

    def refine_to_facet(self, face):
        assert self.is_face(face)
        face = self.subspace().projection(face[1:])
        face = np.hstack((0, face))
        face = scale_to_int(face)
        while True:
            subspace = self.intersection(face).basis()
            nullspace = addz(matrix_nullspace(delz(np.vstack((
                addz(self.subspace().normals),
                subspace,
                face,
            )))))
            try:
                direction = next(self.filter_non_singular_directions(nullspace))
            except StopIteration:
                return face
            face, _ = self.get_adjacent_facet(face, -direction)


class LinearSubspace:

    """
    Utility for vector transformations into a subspace and back.
    """

    def __init__(self, onb, normals):
        self.dim = onb.shape[0]
        self.onb = onb
        self.normals = normals

    @classmethod
    def all_space(cls, dim):
        """Full space."""
        return cls(np.eye(dim), np.empty((0, dim)))

    @classmethod
    def from_rowspace(cls, matrix):
        """Create subspace from basis (row-) vectors."""
        onb, normals = matrix_imker(np.atleast_2d(matrix))
        return cls(onb, normals)

    @classmethod
    def from_nullspace(cls, matrix):
        """Create subspace from normal (row-) vectors."""
        normals, onb = matrix_imker(np.atleast_2d(matrix))
        return cls(onb, normals)

    def nullspace(self):
        """Return the orthogonal complement."""
        return self.__class__(self.normals, self.onb)

    @cached
    def projector(self):
        """Projection matrix onto the subspace."""
        return self.onb.T @ self.onb

    def projection(self, v):
        """Return projection of vector onto the subspace [= back(into(v))]."""
        return v @ self.projector().T

    def into(self, v):
        """Transform outer space vector into subspace."""
        return v @ self.onb.T

    def back(self, v):
        """Transform a subspace vector back to outer space."""
        return v @ self.onb

    def basis_vector(self, i):
        """Get a basis vector of the subspace."""
        return basis_vector(self.dim, i)

    def back_space(self, space):
        """Convert subspace of this space to outer basis."""
        normals = np.vstack((
            self.normals,
            self.back(space.normals),
        ))
        return self.__class__(self.back(space.onb), normals)

    def add_normals(self, normals):
        return self.__class__.from_nullspace(np.vstack((
            self.normals,
            normals,
        )))
