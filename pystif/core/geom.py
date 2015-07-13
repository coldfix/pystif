
import numpy as np

from .array import scale_to_int, make_int_exact
from .linalg import (matrix_rowspace, matrix_nullspace,
                     plane_basis, plane_normal, addz, delz)
from .util import PointSet, cached


class ConvexPolyhedron:

    """
    Utility for finding the extreme points (vertices) of a convex polyhedron
    defined by the projection of a system of linear inequalities.

    Assumes that the origin is an extreme point of the polyhedron.
    """

    def __init__(self, lp, subdim):
        self.lp = lp
        self.subdim = subdim
        self.points = PointSet()

    @classmethod
    def from_cone(cls, system, subdim, limit):
        """Create search problem from the system ``L∙x≥0, x≤limit``."""
        lp = system.lp()
        for i in range(1, subdim):
            lp.set_col_bnds(i, 0, limit)
        return cls(lp, subdim)

    @cached
    def basis(self):
        """
        Return a matrix of extreme points whose convex hull defines an inner
        approximation to the projection of the polytope defined by the LP to
        the subspace of its lowest ``self.subdim`` components.

        Extreme points are returned as matrix rows.

        The returned points define a polytope with the dimension of the
        projected polytope itself (which may be less than ``self.subdim``).
        """
        points = np.empty((0, self.subdim))
        orth = np.eye(self.subdim - 1)
        while orth.shape[1] > 0:
            # Choose vector from orthogonal space and optimize along its
            # direction:
            d = orth[0]
            v = np.dot(d, orth.T)
            v = np.hstack((0, v))
            x = self.search(v)[1:]
            p = make_int_exact(np.dot(x, orth))
            if all(p == 0):
                x = self.search(-v)[1:]
                p = make_int_exact(np.dot(x, orth))
            if all(p == 0):
                # Optimizing along ``v`` yields a vector in our ray space.
                # This means ``v∙x=0`` is part of the LP.
                orth = np.dot(orth, plane_basis(d))
            else:
                # Remove discovered ray from the orthogonal space:
                orth = np.dot(orth, plane_basis(p))
                points = np.vstack((
                    points,
                    np.hstack((0, x)),
                ))
        return points

    @cached
    def onb(self):
        return matrix_rowspace(self.basis())

    def search(self, q):
        """
        Search an extreme point ``x`` of the LP which minimizes ``q∙x``.

        The ``q`` parameter must be specified as a vector with ``subdim``
        components. Its geometric interpretation is the normal vector of a
        hyperplane in the projection space. This hyperplane is shifted along
        its normal until all points inside the polyhedron fulfill ``q∙x≥0``.

        Returns an extreme point with ``subdim`` components (the leftmost
        component is always 0).
        """
        assert len(q) == self.subdim
        assert q[0] == 0
        # For the random vectors it doesn't matter whether we use `minimize`
        # or `maximize` — but it *does* matter for the oriented direction
        # vectors obtained from other functions:
        extreme_point = self.lp.minimize(q, embed=True)
        extreme_point = extreme_point[0:self.subdim]
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
        assert space.shape[1] == self.subdim
        lp = self.lp.copy()
        lp.add(space, 0, 0, embed=True)
        return self.__class__(lp, self.subdim)

    def dim(self):
        """Geometric dimension of the projected polyhedron."""
        return len(self.basis())

    def face_dim(self, face):
        """Return dimension of the face."""
        face = np.dot(face, self.onb().T)
        return self.intersection(face).dim()

    def is_facet(self, face):
        # TODO: need to work with the subspaces bases…?
        return self.face_dim(plane) == self.dim()-1

    def is_face(self, face):
        return self.lp.implies(face, embed=True)

    def get_adjacent_facet(self, face, subface, inner, atol=1e-10):
        """
        Get the adjacent facet defined by `facet` and `subface`.
        """
        plane = -face
        subface = delz(subface)
        while True:
            vertex = self.search(plane)[1:]
            if self.lp.get_objective_value() >= -atol:
                return plane
            # TODO: it should be easy to obtain the result directly from the
            # facet equation, boundary equation and additional vertex without
            # resorting to matrix decomposition techniques.
            plane = plane_normal(np.vstack((subface, vertex)))
            plane = scale_to_int(plane)
            plane = np.hstack((0, plane))
            if np.dot(plane, inner) <= -atol:
                plane = -plane

    def filter_non_singular_directions(self, nullspace):
        for i, direction in enumerate(nullspace):
            if self.is_face(direction):
                direction = -direction
                if self.is_face(direction):
                    continue
            yield i, direction

    def refine_to_facet(self, face):
        subspace = self.intersection(face).basis()
        nullspace = addz(matrix_nullspace(delz(np.vstack((subspace, face)))))
        try:
            i, direction = next(self.filter_non_singular_directions(nullspace))
        except StopIteration:
            return face
        subface = np.vstack((subspace, np.delete(nullspace, i, axis=0)))
        plane = self.get_adjacent_facet(face, subface, direction)
        return self.refine_to_facet(plane)
