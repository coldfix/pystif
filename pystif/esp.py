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

from collections import namedtuple

import numpy as np

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
    # Compute a point on the affine hull of the adjacent facet.
    lp = Problem(P.matrix[r.E])
    b = f.b * (1-eps)
    lp.add_row(f.a, b, b, embed=True)
    x = lp.maximize(r.a, embed=True)

    # Compute the equality set of the adjacent facet.
    E =

    # Compute affine hull of adjacent facet.

    # Normalize and ensure halfspace contains origin.

    # Report adjacent facet.
    return adj


def ridges(P, facet):
    """
    [Algorithm 5.1] Ridge oracle.

    Input:  Polytope P and equality set E such that π(Pᴱ) is a facet of π(P).

    Output: List [Eᵣ] whose elements are all equality sets Eᵣ such that
            π(P_Eᵣ) is a facet of π(Pᴱ).
    """
    pass
    # Initialize variables.
    # Compute the dimension of Pᴱ.
    # Call ESP recursively to compute the ridges.
    # Test each i ∈ Eᶜ to see if E ∪ {i} defines a ridge.
    # Normalize all equations and make orthogonal to the facet π(Pᴱ)


def shoot(P):
    """
    [Algorithm 6.1] Shooting oracle: Calculation of a random facet of π(P).

    Input:  Polytope `P = {(x,y) ∈ Rᵈ⨯ Rᵏ | Cx + Dy ≤ b}`, which contains the
            origin in its interior and whose projection is full-dimensional.

    Output: A randomly selected equality set E₀ of P such that
            `π(P_E₀) ≘ {x | a_f∙x = b_f} ∩ π(P)` is a facet of π(P).
    """
    pass
    # Find a face of P that projects to a facet of π(P)
    # Compute affine hull of facet
    # Handle dual-degeneracy in LP
    # Report facet


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
