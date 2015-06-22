"""
Output elemental inequalities for given number of variables.

Usage:
    makesys -c COLS [-o FILE] INEQ...
    makesys -v VARS [-o FILE] INEQ...
    makesys -v VARS [-o FILE] [INEQ...] -e

Options:
    -o OUTPUT, --output OUTPUT      Write inequalities to this file
    -a, --append                    Append to output file
    -c COLS, --cols COLS            Set column names or count
    -v VARS, --vars VARS            Set variable names or count
    -e, --elem-ineqs                Add elemental inequalities

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
from .core.it import elemental_inequalities, num_vars
from .core.io import (System, SystemFile, _name_list,
                      default_column_labels, column_varname_labels)


def create_index(l):
    return {v: i for i, v in enumerate(l)}


class AutoInsert:

    """Autovivification for column names."""

    def __init__(self, cols):
        self._cols = cols
        self._idx = create_index(cols)

    def __len__(self):
        return len(self._cols)

    def __getitem__(self, key):
        try:
            return self._idx[key]
        except KeyError:
            self._cols.append(key)
            self._idx[key] = index = len(self._cols) - 1
            return index


def _parse_expr(expr):
    re_number = r'\s*([+\-]?(?:\d+(?:\.\d*)?|\d*\.\d+)(?:[eE][+\-]?\d+)?)'
    re_ident = r'\s*([a-zA-Z_][\w\.]*)'
    re_sign = r'\s*([-+])'
    re_head = re.compile("".join((
        '^', re_sign, '?', re_number, '?', re_ident,
        '|^', re_sign, '?', re_number
    )))
    re_tail = re.compile("".join((
        '^', re_sign, re_number, '?', re_ident,
        '|^', re_sign, re_number
    )))
    signs = {'+': 1, '-': -1, None: 1}
    m = re_head.match(expr)
    while m:
        ident = m.group(3)
        if ident:
            sign, number = m.groups()[:2]
        else:
            ident = 0
            sign, number = m.groups()[3:]
        coef = signs[sign]
        if number is not None:
            coef *= float(number)
        yield (ident, coef)
        expr = expr[m.end():]
        m = re_tail.match(expr)
    if expr.strip():
        raise ValueError("Unexpected token at: {!r}".format(expr))


def _parse_eq_line(line, col_idx):
    line = line.strip()
    if not line or line.startswith('#'):
        return []
    m = re.match("^([^≥≤<>=]*)(≤|≥|<=|>=|=)([^≥≤<>=]*)$", line)
    if not m:
        raise ValueError("Invalid constraint format: {!r}.\n"
                         "Must contain exactly one relation".format(line))
    lhs, rel, rhs = m.groups()
    terms = list(_parse_expr(lhs))
    terms += [(col, -coef) for col, coef in _parse_expr(rhs)]
    indexed = [(0 if col == 0 else col_idx[col], coef)
               for col, coef in terms]
    v = np.zeros(len(col_idx))
    for idx, coef in indexed:
        v[idx] += coef
    if rel == '<=' or rel == '≤':
        return [-v]
    if rel == '>=' or rel == '≥':
        return [v]
    return [-v, v]


def _parse_eq_file(eq_str, col_idx):
    if not path.exists(eq_str):
        return _parse_eq_line(eq_str, col_idx)
    with open(eq_str) as f:
        return sum((_parse_eq_line(l, col_idx) for l in f), [])


def main(args=None):
    opts = docopt(__doc__, args)

    if opts['--cols']:
        colnames = _name_list(opts['--cols'])
        if isinstance(colnames, int):
            colnames = default_column_labels(colnames)
        col_idx = create_index(colnames)
    elif opts['--vars']:
        varnames = _name_list(opts['--vars'])
        colnames = column_varname_labels(varnames)
        col_idx = create_index(colnames)
    else:
        colnames = []
        col_idx = AutoInsert(colnames)

    equations = []
    for e in opts['INEQ']:
        equations += _parse_eq_file(e, col_idx)
    dim = len(colnames)
    if opts['--elem-ineqs']:
        equations += list(elemental_inequalities(num_vars(dim)))

    output = SystemFile(opts['--output'], columns=colnames)
    for e in equations:
        output(np.hstack((e, np.zeros(dim-len(e)))))


if __name__ == '__main__':
    main()
