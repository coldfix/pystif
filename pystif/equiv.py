"""
Check the equivalence of two system of equations.

Usage:
    equiv [-q...] [-1] A B

Options:
    -q, --quiet         No output, result is only signaled via return value
    -1, --one-way       Check only if A implies B, not the other way round

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


def check_implies(sys_a, sys_b, name_a, name_b, quiet=False):
    lp = sys_a.lp()
    missing = [constr for constr in sys_b.matrix
               if not lp.implies(constr)]
    if quiet <= 1:
        if missing:
            print("{} does not imply {}!".format(name_a, name_b))
        else:
            print("{} implies {}!".format(name_a, name_b))
    if quiet == 0:
        print("{} misses the following inequalities of {}:"
                .format(name_a, name_b))
        for constr in missing:
            print(format_vector(constr))
    return len(missing) == 0


def main(args=None):
    opts = docopt(__doc__)
    sys_a = System.load(opts['A'])
    sys_b = System.load(opts['B'])
    sys_b, _ = sys_b.slice(sys_a.columns, fill=True)
    status = 0
    kwd = {'quiet': opts['--quiet']}
    if not check_implies(sys_a, sys_b, 'A', 'B', **kwd):
        status |= 1
    if not opts['--one-way']:
        if not check_implies(sys_b, sys_a, 'B', 'A', **kwd):
            status |= 2
    return status


if __name__ == '__main__':
    sys.exit(main())
