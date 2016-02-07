"""
Check the equivalence of two system of equations.

Usage:
    equiv [-q...] [-1] [-y SYMMETRIES] A B

Options:
    -q, --quiet                 Less output, result is signaled via return value
    -1, --one-way               Check only if A implies B, not the other way round
    -y SYM, --symmetry SYM      Symmetry group generators

The program's return code signifies whether the two systems are equivalent:

    0   A and B are equivalent
    1   A does not imply B
    2   B does not imply A
    3   neither A implies B nor B implies A

The values 2 and 3 are only used if --one-way is not in effect.
"""

import sys

import numpy as np
from docopt import docopt

from .core.io import format_vector, System
from .core.symmetry import NoSymmetry, SymmetryGroup, group_by_symmetry


def check_implies(sys_a, sys_b, name_a, name_b, *, symmetries, quiet=False):
    lp = sys_a.lp()
    # take one representative from each category:
    groups = group_by_symmetry(symmetries, sys_b.matrix)
    missing = [g for g in groups if not lp.implies(g[0])]
    if missing:
        if quiet <= 1:
            print("{} misses {} ({} intrinsic) constraints of {}!".format(
                name_a, sum(map(len, missing)), len(missing), name_b))
        if quiet == 0:
            print("{} misses the following inequalities of {}:"
                  .format(name_a, name_b))
            for constr in missing:
                print(format_vector(constr[0]))
        return False
    else:
        if quiet <= 1:
            print("{} implies {}!".format(name_a, name_b))
        return True


def main(args=None):
    opts = docopt(__doc__)
    sys_a = System.load(opts['A'])
    sys_b = System.load(opts['B'])
    sys_b, _ = sys_b.slice(sys_a.columns, fill=True)

    if opts.get('--symmetry'):
        symm = SymmetryGroup.load(opts['--symmetry'], sys_a.columns)
    else:
        symm = NoSymmetry

    status = 0
    kwd = {'quiet': opts['--quiet'], 'symmetries': symm}
    if not check_implies(sys_a, sys_b, 'A', 'B', **kwd):
        status |= 1
    if not opts['--one-way']:
        if not check_implies(sys_b, sys_a, 'B', 'A', **kwd):
            status |= 2
    return status


if __name__ == '__main__':
    sys.exit(main())
