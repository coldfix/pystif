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
# - [2,8] efficient methods for converting between halfspace and vertex
#   representations
# - [4] equality subsystem (similar to equality set)

from collections import namedtuple

import numpy as np
from numpy.linalg import pinv   # Moore-Penrose pseudo-inverse

from .core.app import application
from .core.lp import Problem


# aff F = { π(x) | Lᴱx ≤ mᴱ }
#       = { π(x) | a∙x ≤ b }
Face = namedtuple('Face', [
    'E',    # equality set (list)
    'a',    # normal vector (np.array)
    'b',    # inhomogeneity (float)
])


def esp(P):
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
    L = []
    f = shoot(P)
    for r in ridges(P, f):
        L.append((f, r))
    # Initialize matrix G, vector g and list E.
    R = [f]
    # Search for adjacent facets until the list L is empty.
    while L:
        f = adjacent_facet(P, *L.pop())
        for r in ridges(P, f):
            for i, (_, rr) in enumerate(L):
                if r.E == rr.E:
                    del L[i]
                    break
            else:
                L.append((f, r))
        R.append(f)
    # Report projection.
    return R


# FME:
# - 16
# - 10
# block elimination:
# - 8, 2, 3
# similar:
# - 1


def adjacent_facet(P, f: Face, r: Face, *, eps=1e-7):
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
    b = f.b * (1-eps)
    lp.add_row(f.a, b, b, embed=True)
    xy = lp.maximize(r.a, embed=True)

    # Compute the equality set of the adjacent facet.
    if lp.is_unique():
        E_adj = equality_set(P.matrix[r.E], xy)
    else:
        gamma = (r.f @ x[:d]) / (f.f @ x[:d])
        poly = np.vstack((
            P.matrix[r.E],
            -(f.f + gamma * r.f),
            +(f.f + gamma * r.f),
        ))
        E_adj = equality_set_of_face(poly) # FIXME

    # Compute affine hull of adjacent facet.
    a_adj = matrix_nullspace(D[E_adj].T).T @ C[E_adj]
    # Normalize and ensure halfspace contains origin.
    a_adj /= norm(a_adj)
    # Report adjacent facet.
    return Face(E_adj, a_adj, 0)


def ridges(P, facet):
    """
    [Algorithm 5.1] Ridge oracle.

    Input:  Polytope P and equality set E such that π(Pᴱ) is a facet of π(P).

    Output: List [Eᵣ] whose elements are all equality sets Eᵣ such that
            π(P_Eᵣ) is a facet of π(Pᴱ).
    """
    # Initialize variables.
    E_r = []
    C, D = P.blocks()
    d, k = P.block_dims()
    E_c = sorted(set(range(len(C))) - set(facet.E))
    # Compute S, L, t as in Lemma 31.
    S = C[E_c] - D[E_c] @ pinv(D[facet.E]) @ C[facet.E]
    L = D[E_c] @ matrix_nullspace(D[facet.E])
    t = 0 # P.b[E_c] - D[E_c] @ D[facet.E].H @ b[facet.E]
    C, D = P.blocks()
    # Compute the dimension of Pᴱ.
    if matrix_rank(P.matrix[facet.E]) < k+1:
        # Call ESP recursively to compute the ridges.
        P_f = P.intersection(facet.f)
        E_f = esp(P_f)
        E = [to_ridge(f) for f in E_f] # TODO…
    else:
        # Test each i ∈ Eᶜ to see if E ∪ {i} defines a ridge.
        for i in E_c:
            # Q(i) as defined in Proposition 35
            Q_i = [j for j in E_c
                   if matrix_rank(np.vstack((face.f, S[[i, j]]))) == 2]
            # Compute τ* from LP (17)
            lp = Problem(np.hstack(([1], -S[Q_i])))
            lp.add(np.hstack((0, facet.f)), 0, 0)
            lp.add(np.hstack((0, S[i])), 0, 0)
            lp.add([1], -1, embed=True)
            tau = lp.minimize([1], embed=True)[0]
            if tau < 0 and not np.isclose(tau, 0):
                # TODO: Compute equality set Q(i)??
                E_r.append(Face(Q(i), S[i], 0))

    # Normalize all equations and make orthogonal to the facet π(Pᴱ)
    # TODO…
    return E_r


def shoot(P):
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
        gamma = np.random.normal(size=d)
        trans = np.hstack((C @ gamma, D))
        lp = Problem(trans)
        ry = lp.maximize([1], embed=True)
        E0 = equality_set(trans, ry)        # assumes b=0
        N = matrix_nullspace(D[E0].T)
        f = N.T @ C[E0]
        # TODO: should the condition be rather `dim π(Pᴱ) = d - 1`???
        # until: dim π(P_E₀) = d − rank Nᵀ C_E₀ = d − 1
        if matrix_rank(f) == 1:
            break
    # Compute affine hull of facet
    f /= norm(f)
    # Handle dual-degeneracy in LP
    if lp.is_dual_degenerate():
        # Compute equality set E₀ such that P_E₀ = {(x, y) | a_f∙x = b_f} ∩ P
        E0 = equality_set_of_face(P, f)
    # Report facet
    return Face(E0, f, 0)


def equality_set(matrix, vector):
    """Return list of row indices for which m@v = 0."""
    return np.flatnonzero(np.isclose(matrix @ vector, 0))


def equality_set_of_face(P, f):
    """
    [Appendix A]
    """
    # TODO: this assumes P = {x : Ax ≥ 0}
    lp = P.lp.copy()
    lp.add(f, 0, 0)
    return [i for i, row in enumerate(P.matrix)
            if not np.isclose(lp.maximum(row), 0)]


@application
def main(app):
    system = app.system

    # TODO: translate to origin
    cut = np.hstack((np.ones(app.subdim), np.zeros(app.dim-app.subdim)))

    system.matrix = np.vstack((
        system.matrix,
        -cut,
    ))

    for f in esp(app.polyhedron):
        pass
