"""
Output elemental inequalities for given number of variables.

Usage:
    makesys -c COLS [-o FILE] INEQ...
    makesys -v VARS [-o FILE] INEQ...
    makesys -v VARS [-o FILE] [INEQ...] -b
    makesys -v VARS [-o FILE] [INEQ...] -e

Options:
    -o OUTPUT, --output OUTPUT      Write inequalities to this file
    -a, --append                    Append to output file
    -c COLS, --cols COLS            Set column names or count
    -v VARS, --vars VARS            Set variable names or count
    -e, --elem-ineqs                Add elemental inequalities
    -b, --bell                      Create a bell polytope

This will output a matrix of inequalities for NUM_VARS random variables. Each
row ``q`` corresponds to one inequality

    q∙x ≥ 0.

The columns correspond to the entropies of

    ∅, X₀, X₁, X₀X₁, X₂, X₀X₂, X₁X₂, X₀X₁X₂, …

and so on, i.e. the bit representation of the column index corresponds to the
subset of variables. The zero-th column will always be zero.
"""

import re
import sys
from os import path

from docopt import docopt
import numpy as np

from .core.it import elemental_inequalities, num_vars, bits_to_num
from .core.io import (System, SystemFile, _name_list, get_bits, supersets,
                      default_column_labels, column_varname_labels)

from .core.parse import parse_files, to_numpy_array


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


def vstack(a, b):
    return np.vstack((a, list(b)))


def main(args=None):
    opts = docopt(__doc__, args)

    if opts['--cols']:
        colnames = _name_list(opts['--cols'])
        if isinstance(colnames, int):
            colnames = default_column_labels(colnames)
    elif opts['--vars']:
        varnames = _name_list(opts['--vars'])
        colnames = column_varname_labels(varnames)
    else:
        colnames = []

    equations, colnames = to_numpy_array(parse_files(opts['INEQ']), colnames)

    dim = len(colnames)
    if opts['--elem-ineqs']:
        equations = vstack(equations, elemental_inequalities(num_vars(dim)))
    elif opts['--bell']:
        equations = vstack(equations, map(p_to_q, positivity(dim)))
        # TODO: also normalization

    output = SystemFile(opts['--output'], columns=colnames)
    for e in equations:
        output(np.hstack((e, np.zeros(dim-len(e)))))


if __name__ == '__main__':
    main()
