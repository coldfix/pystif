"""
Convert a system of linear constraints from human readable expressions to a
simple matrix format compatible with ``numpy.loadtxt``.

Usage:
    makesys [-o OUTPUT] INPUT...
    makesys [-o OUTPUT] -b VARS

Options:
    -o OUTPUT, --output OUTPUT      Write inequalities to this file
    -b VARS, --bell VARS            Output bell polytope in Q space

The positional arguments can either be filenames or valid input expressions.

Each row ``q`` of the output matrix corresponds to one inequality

    q∙x ≥ 0.

The columns correspond to the entropies of

    ∅, X₀, X₁, X₀X₁, X₂, X₀X₂, X₁X₂, X₀X₁X₂, …

and so on, i.e. the bit representation of the column index corresponds to the
subset of variables. The zero-th column will always be zero.
"""

from docopt import docopt

import numpy as np

from .core.io import SystemFile, _name_list, get_bits, supersets
from .core.it import num_vars, bits_to_num
from .core.parse import parse_files


def _column_label(index, varnames="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    return "".join(varnames[i] for i in get_bits(index))


def column_varname_labels(varnames):
    if isinstance(varnames, int):
        varnames = [chr(ord('A') + i) for i in range(varnames)]
    dim = 2**len(varnames)
    return ['_'] + [_column_label(i, varnames) for i in range(1, dim)]


def p_to_q(p_vec):
    """
    Transform a probability vector from Q to P parametrization.

    Consider a collection of n probability variables X1,…,Xn that take values
    in {0, 1}. Any realization of X1,…,Xn can be represented as a bit string
    of n digits.

    For a subset A ⊂ {1,…,n} we define p and q by

        q(A) = Prob{ (Xi=1 ∀ i in A) }
        p(A) = Prob{ (Xi=1 ∀ i in A) AND (Xj=0 ∀ i not in A) }

    In words:

        q(A) is the probability that "at least" all the variables indexed by
        elements of A assume the value 1 – while all the others may be 0 or 1.

        p(A) is the probability that "exactly" all the variables indexed by
        elements of A assume the value 1 – while all the others must be 0.

    Thus q is related to p by the transformation:

        q(A) = Σ[B ⊃ A] p(B)

    This relation can be inverted with the Möbius inversion forumla to give:

        p(A) = Σ[B ⊃ A] q(B) × (-1)**(|A| + |B|)

    Assuming the input argument is a vector  v = Σ v_A p(A)  the result is
    computed by substiting above inversion formula for p(A) and collecting
    terms with same q(B).
    """
    b_len = num_vars(len(p_vec))
    q_vec = np.zeros(p_vec.shape)
    total = set(range(b_len))
    for i, v in enumerate(p_vec):
        sub = get_bits(i)
        for sup in supersets(sub, total):
            sign = (-1) ** (len(sup) + len(sub))
            q_vec[bits_to_num(sup)] += v * sign
    return q_vec


def q_to_p(q_vec):
    """
    This function transforms a probability vector from Q to P parametrization.

    The input argument is a vector  v = Σ v_A q(A)  and the result is computed
    by substiting

        q(A) = Σ[B ⊇ A] p(B)

    and collecting the terms of p(B) for same B in the corresponding component.

    I implemented this function mainly as a consistency check for the `p_to_q`
    function above. They should be the inverse of each other, i.e.

        >>> x == q_to_p(p_to_q(x))
        True
    """
    b_len = num_vars(len(q_vec))
    p_vec = np.zeros(q_vec.shape)
    total = set(range(b_len))
    for i, v in enumerate(q_vec):
        sub = get_bits(i)
        for sup in supersets(sub, total):
            p_vec[bits_to_num(sup)] += v
    return p_vec


def main(args=None):
    opts = docopt(__doc__, args)

    if opts['--bell']:
        varnames = _name_list(opts['--bell'])
        colnames = column_varname_labels(varnames)
        dim = len(colnames)
        equations = np.vstack(map(p_to_q, np.eye(dim)[1:]))
        symmetries = []
        # equations = np.vstack(map(q_to_p, equations))

    else:
        equations, colnames, symmetries = parse_files(opts['INPUT'])

    output = SystemFile(opts['--output'], columns=colnames,
                        symmetries=symmetries)
    for e in equations:
        output(e)


if __name__ == '__main__':
    main()
