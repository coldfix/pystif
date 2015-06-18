# encoding: utf-8
"""
Exports functions that are specific to the interpretation of vectors as
inequalities on the set of joint entropies. The correspondence is as follows:

Identify each variable with its index i from I = {0, 1, …, N-1}. Then entropy
is a real valued set function from the power set of indices P = 2ᴵ and can be
represented as a vector of dimension 2ᴺ. The component s corresponds to the
joint entropy H(S) of a subset S ⊂ I where the i-th variable is in S iff the
i-th bit is non-zero in s, e.g.:

        13 = 1101₂ ~ H(X₀,X₂,X₃)

The zero-th vector component corresponds to the entropy of the empty set which
is defined to be zero. It is not removed from the vector to avoid off-by-one
mistakes. In fact, this matches up nicely with the convention used in pystif
that the zero-th column is used for the inhomogeneity of an inequality (which
is of course zero for elemental inequalities).
"""

import numpy as np
import math


def _insert_zero_bit(pool, bit_index):
    """Insert a zero bit at specified position."""
    bit = 1<<bit_index
    left = (pool & ~(bit-1)) << 1
    right = pool & (bit-1)
    return left | right


def _nCr(n, r):
    """Binomial coefficient (n choose r)."""
    fac = math.factorial
    return fac(n) // fac(r) // fac(n-r)


def num_vars(dim):
    """Get the number of random variables from the entropy space dimension."""
    num_vars = int(round(math.log2(dim)))
    assert 2**num_vars == dim
    return num_vars


def num_elemental_inequalities(num_vars):
    """Get number of elemental inequalities given number of variables."""
    return num_vars + _nCr(num_vars, 2) * 2**(num_vars-2)


def elemental_inequalities(num_vars, dtype=np.float64):
    """Iterate all elemental inequalities for specified variable count."""
    if num_vars < 1:
        raise ValueError("Invalid number of variables: {}".format(num_vars))
    if num_vars == 1:
        yield np.array([0, 1])
        return
    dim = 2**num_vars
    sub = 2**(num_vars-2)
    All = dim-1
    # Iterate all elemental conditional entropy positivities, i.e. those of
    # the form H(a|A) ≥ 0 where A = {i ≠ a}:
    for a in range(num_vars):
        A = All ^ (2**a)
        row = np.zeros(dim, dtype)
        row[A  ] = -1
        row[All] = +1
        yield row
    # Iterate all elemental conditional mutual information positivities, i.e.
    # those of the form H(a:b|K)>=0 where a,b not in K:
    for a in range(num_vars-1):
        for b in range(a+1, num_vars):
            A = 2**a
            B = 2**b
            for i in range(sub):
                K = _insert_zero_bit(_insert_zero_bit(i, a), b)
                row = np.zeros(dim, dtype)
                row[A  |K] = +1
                row[  B|K] = +1
                row[A|B|K] = -1
                if K:
                    row[K] = -1
                yield row


def cyclic_cca_causal_constraints_1d(width, dtype=np.float64):
    """
    Iterate causal constraints in first layer of a CCA of the given width.

    Each constraint is a conditional independency which is returned as the
    vector of its coefficients for the joint entropies.

    The structure of the CCA is assumed to be hexagonal, i.e. each cell is
    influenced by two parent cells, like in the following picture:

        A₀  A₁  A₂  A₃
          B₀  B₁  B₂  B₃
    """
    dim = 2**(2*width)
    All = dim-1
    for left in range(width):
        right = (left+1) % width
        Var = 2**(left)
        Pa = 2**(width+left) | 2**(width+right)
        Nd = All ^ (Var | Pa)
        row = np.zeros(dim, dtype)
        row[Pa|Var] = +1
        row[Pa|Nd]  = +1
        row[Pa]     = -1
        row[All]    = -1
        yield row


def mutual_independence_constraints(cell_nos, num_vars_tot, dtype=np.float64):
    """Iterate mutual independence constraints for the given set of cells."""
    # Currently, only return a single constraint c that can be used either as
    # inequality c∙x ≥ 0 or as equality c∙x = 0:
    dim = 2**num_vars_tot
    row = np.zeros(dim, dtype)
    joint = 0
    for cell_no in cell_nos:
        joint |= 2**cell_no
        row[2**cell_no] = -1
    row[joint] = +1
    yield row
