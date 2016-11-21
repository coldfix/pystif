"""
Find projection of a convex cone to a lower dimensional subspace using the
Equality Set Projection (ESP) algorithm described in

    Equality Set Projection: A new algorithm for the projection of polytopes
    in halfspace representation – Colin N. Jones and Eric C. Kerrigan and Jan
    M. Maciejowski, 2004

Usage:
    esp INPUT -s SUBSPACE [-o OUTPUT] [-l LIMIT] [-y SYMMETRIES] [-q]... [-v]... [-p] [-i FILE]

Options:
    -o OUTPUT, --output OUTPUT      Save facets of projected cone
    -s SUB, --subspace SUB          Subspace specification (dimension or file)
    -l LIMIT, --limit LIMIT         Add constraints H(i)≤LIMIT for i<SUBDIM
                                    [default: 1]
    -y SYM, --symmetry SYM          Symmetry group generators
    -q, --quiet                     Less status output
    -v, --verbose                   Show more output
    -p, --pretty                    Pretty print output inequalities
    -i FILE, --info FILE            Print short summary to file (YAML)
"""

# Interesting references in the ESP paper:
# - [2,8]   efficient methods for converting between H-/V-representation
# - [4]     equality subsystem (similar to equality set)
# - [10,16] FME
# - [2,3,8] block elimination
# - [1]     similar algorithm

# TODO:
# - change inequality signs to `≥` (+fix optimization directions)
# - work on convex cones
# - check whether ESP needs x≥0 as part of the matrix description

import numpy as np
from numpy.linalg import pinv, norm     # Moore-Penrose pseudo-inverse
from numpy import sign

from .core.linalg import (
    matrix_rank as rank,
    random_direction_vector,
    matrix_nullspace,
)
from .core.app import application
from .core.lp import Problem as _Problem
from .core.geom import ConvexCone, unit_vector


inf = float("inf")

def Problem(matrix):
    lp = _Problem(matrix, lb_row=-inf, ub_row=0)
    lp.set_col_bnds(0, 1, 1)
    return lp


# NOTE: we reinterpret the ConvexCone `C = {x : Lx ≥ 0}` as a polytope
# `P = {y : Ay ≤ b}`, by the identification `y = (1, x)`, `L = (-b | -A)`:
Polytope = ConvexCone


def vstack(*args): return np.vstack(args)
def hstack(*args): return np.hstack(args)


def null(A):
    return matrix_nullspace(A).T


class Face:

    def __init__(self, E, a):
        self.E = E      # equality set (list)
        self.a = a      # normal vector (np.array)

    @property
    def E(self):
        return self._E_list

    @E.setter
    def E(self, E):
        self._E_list = sorted(E)
        self._E_set = frozenset(self._E_list)

    def __hash__(self):
        return hash(self._E_set)

    def __eq__(self, other):
        return self.E_set == other.E_set


def esp(P: Polytope):
    """Full ESP algorithm. See :func:`esp_worker` for more details."""
    P, t1 = _at_origin(P)
    P, t2 = _with_full_rank(P)
    facets = esp_worker(P)
    return [Face(facet.E, t1(t2(facet.a)))
            for facet in facets]


def esp_worker(P: Polytope):
    """
    [Algorithm 3.1] Equality Set Projection ESP.

    Input:  Polytope P = {x | Lx ≤ m} whose projection is full-dimensional
            and contains the origin in its interior.

    Output: Matrix G and vector g such that {x ∈ Rᵈ | Gx ≤ g} is an
            irredundant description of π(P). List E of all equality sets E of
            P such that π(Pᴱ) is a facet of π(P).
            Given as list of :class:`Face`.
    """
    # Initialize ridge-facet list L with random facet.
    f = shoot(P)
    L = {r: f for r in ridges(P, f)}
    # Initialize matrix G, vector g and list E.
    R = [f]
    # Search for adjacent facets until the list L is empty.
    while L:
        f = adjacent_facet(P, *L.popitem())
        for r in ridges(P, f):
            try:
                L.pop(r)
            except KeyError:
                L[r] = f
        R.append(f)
    # Report projection.
    return R


def adjacent_facet(P: Polytope, r: Face, f: Face, *, eps=1e-7):
    """
    [Algorithm 4.1] Adjacency oracle.

    Input:  Polytope P and equality sets Eᵣ and E such that π(P_Eᵣ) is a
            ridge and π(Pᴱ) is a facet of π(Pᴱ) and Eᵣ⊃ E.
            Orthogonal unit vectors a_f and a_r and scalars b_f and b_r such
            that `aff π(Pᴱ) = {x | a_f∙x = b_f}`
            and `aff π(P_Eᵣ) = {x | a_r∙x = b_r} ∩ aff π (Pᴱ)`.

    Output: Equality set E_adj such that π(Pᴱ_adj) is a facet of π(P) and
            π(P_Eᵣ)⊂ π(Pᴱ_adj).
    """
    d, k = P.block_dims()
    # Compute a point on the affine hull of the adjacent facet.
    lp = Problem(P.matrix[r.E])
    a_f = hstack(f.a[0]*(1-eps), f.a[1:])
    lp.add_row(a_f, 0, 0, embed=True)
    xy = lp.maximize(r.a, embed=True)

    # Compute the equality set of the adjacent facet.
    M_r = P.matrix[r.E]
    if lp.is_unique():
        E_adj_r = active_constraints(M_r, xy)
    else:
        gamma = - (r.a @ x[:d]) / (f.a @ x[:d])
        face = f.a + gamma * r.a
        E_adj_r = equality_set(P, face, M_r)
    E_adj = np.array(r.E)[E_adj_r]

    # Compute affine hull of adjacent facet.
    a_adj = null(D[E_adj].T).T @ C[E_adj]
    a_adj = norm_ineq(a_adj)
    # Report adjacent facet.
    return Face(E_adj, a_adj)


def ridges(P: Polytope, facet: Face):
    """
    [Algorithm 5.1] Ridge oracle.

    Input:  Polytope P and equality set E such that π(Pᴱ) is a facet of π(P).

    Output: List [Eᵣ] whose elements are all equality sets Eᵣ such that
            π(P_Eᵣ) is a facet of π(Pᴱ).
    """
    # Initialize variables.
    C, D = P.blocks()
    d, k = P.block_dims()
    E = facet.E
    E_c = sorted(set(range(len(C))) - set(E))
    # Compute S, L, t as in Lemma 31.
    S = C[E_c] - D[E_c] @ pinv(D[E]) @ C[E]
    L = D[E_c] @ null(D[E])
    # Compute the dimension of Pᴱ.
    if rank(P.matrix[E]) < k+1:
        # Call ESP recursively to compute the ridges.
        P_f = P.activate(E)
        E_f = esp(P_f)
    else:
        # Test each i ∈ Eᶜ to see if E ∪ {i} defines a ridge.
        def Q(S_i):         # Q(i) as defined in Proposition 35
            return [S_j for S_j in S if rank(vstack(facet.a, S_i, S_j)) == 2]
        def LP17(S_i, S_Qi, c=1):
            lp = Problem(S_Qi)
            lp.add(facet.a, 0, 0)
            lp.add(S_i, 0, 0)
            i_tau = lp.add_col(-np.ones(len(S_Qi)), -c, embed=True)
            return lp.minimize(unit_vector(i_tau+1, i_tau))[-1]
        E_f = [
            Face(Q_i, S_i)
            for S_i in S
            for Q_i in [Q(S_i)]
            for tau in [LP17(S_i, Q_i)]
            if tau < 0 and not np.isclose(tau, 0)
        ]
    # Normalize all equations and make orthogonal to the facet π(Pᴱ)
    return [Face(ridge.E, to_ridge(ridge.a, facet.a)) for ridge in E_f]


def to_ridge(ridge: np.array, facet: np.array):
    """
    [Remark 36] Convert the facets of π(F) into ridges of π(P), i.e. make
    orthogonal to the facet π(Pᴱ).
    """
    facet = norm_ineq(facet)
    ridge = ridge - (ridge @ facet) * facet
    return norm_ineq(ridge)


def shoot(P: Polytope):
    """
    [Algorithm 6.1] Shooting oracle: Calculation of a random facet of π(P).

    Input:  Polytope `P = {(x,y) ∈ Rᵈ⨯ Rᵏ | Cx + Dy ≤ b}`, which contains the
            origin in its interior and whose projection is full-dimensional.

    Output: A randomly selected equality set E₀ of P such that
            `π(P_E₀) ≘ {x | a_f∙x = b_f} ∩ π(P)` is a facet of π(P).
    """
    C, D = P.blocks()
    d, k = P.block_dims()
    # Find a face of P that projects to a facet of π(P)
    while True:
        gamma = random_direction_vector(d-1).reshape((d-1, 1))
        trans = hstack(C[:,:1], C[:,1:] @ gamma, D)
        lp = Problem(trans)
        ry = lp.maximize(unit_vector(k+2, 1))
        E0 = active_constraints(trans, ry)        # assumes b=0
        # Compute affine hull of facet
        N = null(D[E0].T)
        a = N.T @ C[E0]
        # TODO: should the condition be rather `dim π(Pᴱ) = d - 1`???
        # until: dim π(P_E₀) = d − rank Nᵀ C_E₀ = d − 1
        if rank(a[:,1:]) == 1:
            break
    a = norm_ineq(a[0])
    # Handle dual-degeneracy in LP
    if lp.is_dual_degenerate():
        # Compute equality set E₀ such that P_E₀ = {(x, y) | a_f∙x = b_f} ∩ P
        E0 = equality_set(P, a)
    # Report facet
    return Face(E0, a)


def norm_ineq(a: np.array):
    """Normalize and ensure halfspace contains origin."""
    return a * (sign(a[0]) /
                norm(a[1:]))


def active_constraints(matrix, point):
    """Return list of row indices for which m@v = 0."""
    return np.flatnonzero(np.isclose(matrix @ point, 0))


def equality_set(P: Polytope, a: np.array = None, M: np.array = None):
    """[Appendix A] Calculation of the Affine Hull."""
    # NOTE: this assumes P = {x : Mx ≤ 0}
    lp = P.lp
    if a is not None:
        lp = lp.copy()
        lp.add(a, 0, 0, embed=True)
    if M is None:
        M = P.matrix
    return [
        i for i, row in enumerate(M)
        if np.isclose(lp.minimum(row), 0)
    ]


def _with_full_rank(P):
    """[Appendix B] Projection of non Full-Dimensional Polytopes."""
    C_, D = P.blocks()
    # find subspace in which P is contained
    A = equality_set(P)
    if not A:
        return (P, lambda a: a)
    F_ = null(D[A].T).T @ C_[A]
    b, C = C_[:,:1], C_[:,1:]
    f, F = F_[:,:1], F_[:,1:]
    N_F = null(F)
    M = hstack(- b + C @ pinv(F) @ f, C @ N_F, D)
    lp = Problem(M)
    # apply ESP
    dim = C_.shape[1] - rank(F)
    return (Polytope(lp, dim),
            lambda a: hstack(a[0], a[1:] @ N_F.T))


def _at_origin(P):
    """[Appendix C] Projection of Polytopes that do not Contain the Origin."""
    # translate polytope to the origin
    d, k = P.block_dims()
    lp = P.lp.copy()
    i_tau = lp.add_col(-1*(lp.get_row_lbs() != 0))
    translate = lp.minimize(unit_vector(d+k+1, i_tau))[:-1]
    M = P.matrix
    lp = P.lp.copy()
    lp.set_col(0, M @ translate)
    back = hstack(1, -translate[1:d])
    return (Polytope(lp, P.dim),
            lambda a: hstack(a @ back, a[1:]))


@application
def main(app):
    system = app.system

    # make sure there is a const-column
    if '_' in system.columns:
        system, _ = system.prepare_for_projection('_')
        matrix = system.matrix
    else:
        system.columns = ['_'] + system.columns
        matrix = hstack(
            np.zeros((len(system.matrix), 1)),
            system.matrix)

    num_rows, num_cols = matrix.shape

    matrix = vstack(
        # change inequalities from `a∙x ≥ b` to `-a∙x ≤ -b`:
        -matrix,
        # make lower bounds on columns xₖ≥0 explicit
        hstack(np.zeros((num_cols-1, 1)), -np.eye(num_cols-1)),
        # add a bound Σxₖ≤1
        hstack(-1, np.ones(num_cols-1)),
    )

    lp = Problem(matrix)

    polytope = Polytope(lp, system.subdim)

    # write updated system back
    system.matrix = matrix
    app.system = system
    app.polyhedron = polytope

    for f in esp(polytope):
        pass
