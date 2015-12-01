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
    b_len = num_vars(len(p_vec))
    q_vec = np.zeros(p_vec.shape)
    total = set(range(b_len))
    for i, v in enumerate(p_vec):
        sub = get_bits(i)
        for sup in supersets(sub, total):
            sign = (-1) ** (len(sub) + len(sup) - b_len)
            q_vec[bits_to_num(sup)] += v * sign
    return q_vec


def positivity(dim):
    for i in range(1, dim):
        x = np.zeros(dim)
        x[i] = 1
        yield x


def main(args=None):
    opts = docopt(__doc__, args)

    if opts['--bell']:
        varnames = _name_list(opts['--bell'])
        colnames = column_varname_labels(varnames)
        dim = len(colnames)
        equations = np.vstack(map(p_to_q, positivity(dim)))

    else:
        equations, colnames = parse_files(opts['INPUT'])

    output = SystemFile(opts['--output'], columns=colnames)
    for e in equations:
        output(e)


if __name__ == '__main__':
    main()
