
import numpy as np

from .array import scale_to_int, make_int_exact
from .linalg import (matrix_imker_nice, matrix_nullspace, matrix_rowspace,
                     random_direction_vector, basis_vector, plane_normal,
                     as_column_vector)
from .lp import Problem
from .util import PointSet, cached


def unit_vector(dim, idx):
    r = np.zeros(dim)
    r[idx] = 1
    return r


class ConvexCone:

    """
    Utility for finding the extreme points (vertices) of a convex polyhedron
    defined by the projection of a system of linear inequalities.

    Assumes that the origin is an extreme point of the polyhedron.
    """

    def __init__(self, lp, dim):
        self.lp = lp
        self.dim = dim
        self.points = PointSet()
        self.matrix = lp.get_matrix()

    def blocks(self):
        return np.hsplit(self.matrix, [self.dim])

    def block_dims(self):
        return self.dim, self.matrix.shape[1] - self.dim

    @classmethod
    def from_cone(cls, system, dim, limit):
        """Create search problem from the system ``L∙x≥0, x≤limit``."""
        lp = system.lp()
        for i in range(dim):
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
        orth = LinearSubspace.all_space(self.dim)
        while orth.dim > 0:
            # Choose vector from orthogonal space and optimize along its
            # direction:
            d = random_direction_vector(orth.dim)
            v = orth.back(d)
            x = self.search(v)
            p = make_int_exact(orth.into(x))
            if all(p == 0):
                x = self.search(-v)
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
                    x
                ))
        return points

    @cached
    def subspace(self):
        return LinearSubspace.from_rowspace(self.basis())

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
        extreme_point = scale_to_int(extreme_point)
        self.points.add(extreme_point)
        return extreme_point

    def search_ensure_vertex(self, q):
        """
        Like ``search()``, but ensure that the resulting point is an extreme
        point.

        This uses the extreme_point algorithm in 5.4.1. from

            Lassez, C. and Lassez, J.L., 1990, Quantifier Elimination for
            Conjunctions of Linear Constraints via a Convex Hull Algorithm
        """
        dim = self.dim
        m = np.empty(dim)
        assert len(q) == dim
        k = next(k for k, qk in enumerate(q) if qk != 0)
        inf = float("inf")

        lp = self.lp.copy()
        h = lp.minimum(q, embed=True)

        # replace x_k by (h - Σa[i] x[i]) in Y
        for i, row in enumerate(lp.get_matrix()):
            c = row[k]/q[k]
            if row[k] == 0:
                continue
            lb, ub = lp.get_row_bnds(i)
            for j in range(dim):
                row[j] -= q[j]*c
            lp.set_row(i, row)
            lp.set_row_bnds(i, lb-h*c, ub-h*c)

        for i in range(dim):
            if i == k:
                continue
            m[i] = lp.minimum(unit_vector(dim, i), embed=True)
            # replace x[i] by its optimum m[i] in Y
            lp.set_col_bnds(i, m[i], m[i])

        m[k] = (h - sum(q[i]*m[i] for i in range(dim) if i != k)) / q[k]

        extreme_point = scale_to_int(m)
        self.points.add(extreme_point)
        return extreme_point

    # The above variant sometimes fails (don't ask me why…), use my own
    # (slightly more inefficient) variant instead:
    def search_ensure_vertex(self, q):
        """
        Like ``search()``, but ensure that the resulting point is an extreme
        point. For unknown reasons (numerical instabilities?) this sometimes
        fails with pystif.core.lp.NoFeasibleError.
        """
        assert len(q) == self.dim
        lp = self.lp.copy()
        onb = matrix_rowspace(np.vstack((q, np.eye(self.dim))))
        for q_i in onb:
            v = lp.minimum(q_i, embed=True)
            lp.add_row(q_i, v, v, embed=True)

        extreme_point = lp.get_prim_solution()
        extreme_point = extreme_point[0:self.dim]
        extreme_point = scale_to_int(extreme_point)
        return extreme_point

    def intersection(self, space):
        """
        Return the ConvexCone obtained from the intersection with the
        given subspace.

        :param space: specified by its normal vector(s).
        """
        space = np.atleast_2d(space)
        assert space.shape[1] == self.dim
        lp = self.lp.copy()
        lp.add(space, 0, 0, embed=True)
        return self.__class__(lp, self.dim)

    def activate(self, rows):
        """
        Return the ConvexCone obtained by replacing the inequalities with the
        specified indices with equalities.

        :param frozenset rows:
        """
        lp = self.lp.copy()
        inf = float("inf")
        for row in rows:
            lb, ub = lp.get_row_bnds(row)
            has_lb = lb > -inf
            has_ub = ub < +inf
            if has_lb and not has_lb:
                lp.set_row_bnds(row, lb, lb)
            elif has_ub and not has_lb:
                lp.set_row_bnds(row, ub, ub)
                m = lp.minimum(lp.get_row(row))
            elif lb != ub:
                # free or double bounded row:
                raise NotImplementedError
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
                sx * plane - fx * inner,
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

    def refine_to_facet(self, face, outer_point=None, eps=1e-12):
        assert self.is_face(face)
        face = self.subspace().projection(face)
        face = scale_to_int(face)
        while True:
            subspace = self.intersection(face).basis()
            nullspace = matrix_nullspace(np.vstack((
                self.subspace().normals,
                subspace,
                face,
            )))
            try:
                direction = next(self.filter_non_singular_directions(nullspace))
            except StopIteration:
                return face
            face, subface = self.get_adjacent_facet(face, -direction)
            if outer_point is not None and face @ outer_point > eps:
                face, _ = self.get_adjacent_facet(face, subface)

    def face_to_facets(self, face, eps=1e-12):
        """Convert a face to a list of facets that imply the given face."""
        lp = Problem([face], lb_row=-1)
        while lp.minimum(face) < -eps:
            x = lp.get_prim_solution()
            f = self.refine_to_facet(face, x)
            lp.add(f)
            yield f

    def nullspace_int(self) -> list:
        """
        Return basis vectors of nullspace – but with integer coefficients
        rather than whatever comes out of the nullspace analysis.

        NOTE: This is a slow operation… If you want performance, just use the
        subspace().normals matrix.
        """
        subspace = self.subspace()
        normals = subspace.normals
        if len(normals) == 0:
            return []
        if len(normals) == 1:
            return [scale_to_int(normals[0])]
        # Let's do some magic to get integer coefficients (this is slow…):
        L = self.lp.get_matrix().T
        num_cols = L.shape[1]
        lp = Problem(
            num_cols=num_cols,                      # Find f = qL s.t.
            lb_col=0)                               #    q_i ≥ 0  ∀ i
        lp.add(np.ones(num_cols), 1, 1)             #   Σq_i = 1
        lp.add(L[self.dim:], 0, 0)                  # (qL)_i = 0  ∀ i > m
        lp.add(subspace.onb @ L[:self.dim], 0, 0,   #    qLP = 0
               embed=True)
        normals = []
        orth = subspace.nullspace()
        while orth.dim > 0:
            # Choose vector from orthogonal space and optimize along its
            # direction:
            d = random_direction_vector(orth.dim)
            v = orth.back(d)
            q = lp.minimize(v @ L[:self.dim], embed=True)
            q = scale_to_int(q)
            f = (L @ q)[:self.dim]
            p = make_int_exact(orth.into(f))
            new = orth.back_space(LinearSubspace.from_nullspace(p))
            # TODO: This condition should always be fulfilled by the
            # construction of f – but for now my trust is limited:
            if new.dim < orth.dim and self.is_face(f) and self.is_face(-f):
                orth = new
                normals.append(f)
        return normals


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
        onb, normals = matrix_imker_nice(np.atleast_2d(matrix))
        return cls(onb, normals)

    @classmethod
    def from_nullspace(cls, matrix):
        """Create subspace from normal (row-) vectors."""
        normals, onb = matrix_imker_nice(np.atleast_2d(matrix))
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

    def contains(self, point):
        """Check if a point is element of the subspace."""
        return np.allclose(point @ self.normals.T, 0)
