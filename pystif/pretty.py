"""
Print equations in human readable format.

Usage:
    pretty -i INPUT [-o OUTPUT] [-c] [-g]

Options:
    -i INPUT, --input INPUT         Input file
    -o OUTPUT, --output OUTPUT      Output file
    -c, --canonical                 Assume canonical column labels
    -g, --group                     Group similar constraints
"""


from collections import Counter, defaultdict
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
    return "{} ≥ {}".format(" ".join(lhs).lstrip('+ '), _fmt_float(rhs))


def vars_from_colname(column_name):
    # TODO: more sophisticated parsing?
    return frozenset(column_name)


def character(vals, varsets, allvars):
    """
    Return a value that characterizes the constraint 'vals' in the sense that
    constraints that are permutations of each other are guaranteed to have the
    same character.

    However, it is not guaranteed that constraints with the same character are
    indeed permutations of each other.
    """
    return frozenset(Counter([
        frozenset((coef, len(varsets[i]))
                  for i, coef in enumerate(vals)
                  if coef != 0 and var in varsets[i])
        for var in allvars
    ]).items())


def group_permuted_terms(constraints, colnames):
    varsets = [frozenset()] + [vars_from_colname(n) for n in colnames]
    allvars = frozenset.union(*varsets)
    by_character = defaultdict(list)
    for c in constraints:
        by_character[character(c, varsets, allvars)].append(c)
    return by_character.values()


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
    def dump(rows):
        for row in rows:
            print_(format_human_readable(row, columns))
    if opts['--group']:
        first = True
        for g in group_permuted_terms(system.matrix, columns):
            first = not first and print_()
            dump(g)
    else:
        dump(system.matrix)


if __name__ == '__main__':
    main()
