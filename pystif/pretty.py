"""
Print equations in human readable format.

Usage:
    pretty -i INPUT [-o OUTPUT] [-c]

Options:
    -i INPUT, --input INPUT         Input file
    -o OUTPUT, --output OUTPUT      Output file
    -c, --canonical                 Assume canonical column labels
"""


from math import log2
from docopt import docopt
from .core.it import num_vars
from .util import (System, print_to,
                   default_column_labels, column_varname_labels)


def _fmt_float(f):
    if round(f) == f:
        return str(int(f))
    return str(f)


def _coef(coef):
    if coef < 0:
        prefix = '-'
        coef = -coef
    else:
        prefix = '+'
    if coef != 1:
        prefix += ' ' + _fmt_float(coef)
    return prefix


def format_human_readable(constraint, columns):
    lhs = [(_coef(c), columns[i], c > 0)
           for i, c in enumerate(constraint[1:])
           if c != 0]
    # len() is used as approximation for number of terms involved. For most
    # cases this should be fine.
    lhs = sorted(lhs, key=lambda term: (term[2], len(term[1])),
                 reverse=True)
    lhs = ["{} {}".format(coef, col) for coef, col, _ in lhs]
    if not lhs:
        lhs = ["0"]
    rhs = -constraint[0]
    return "{} ≥ {}".format(" ".join(lhs), _fmt_float(rhs))


def main(args=None):
    opts = docopt(__doc__, args)
    system = System.load(opts['--input'])
    if system.columns:
        columns = system.columns[1:]
    elif opts['--canonical']:
        columns = column_varname_labels(num_vars(system.dim))[1:]
    else:
        columns = default_column_labels(system.dim)
    print_ = print_to(opts['--output'])
    for row in system.matrix:
        print_(format_human_readable(row, columns))


if __name__ == '__main__':
    main()
