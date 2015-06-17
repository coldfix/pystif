"""
Check the equivalence of two system of equations.

Usage:
    equiv [-e] [-q] [-1] A B

Options:
    -e, --elem-ineqs    Add elemental inequalities before checking difference
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
from .core.it import elemental_inequalities, num_vars
from .util import format_vector, System


def show_missing_constraints(lp, constraints, name_a, name_b):
    """Show the difference of these constraints."""
    missing = [constr for constr in constraints.matrix
               if not lp.implies(constr)]
    if not missing:
        print("{} implies {}!".format(name_a, name_b))
        return 0
    print("{} does not imply the following inequalities of {}:"
          .format(name_a, name_b))
    for constr in missing:
        print(format_vector(constr))
    return len(missing)


def check_implies(sys_a, sys_b, name_a, name_b, elem_ineqs=False, quiet=False):
    lp = sys_a.lp()
    if elem_ineqs:
        lp.add(list(elemental_inequalities(num_vars(lp.num_cols))))
    if quiet:
        return lp.implies(sys_b.matrix)
    return show_missing_constraints(lp, sys_b, name_a, name_b) == 0


def main(args=None):
    opts = docopt(__doc__)
    sys_a = System.load(opts['A'])
    sys_b = System.load(opts['B'])
    status = 0
    kwd = {'elem_ineqs': opts['--elem-ineqs'],
           'quiet': opts['--quiet']}
    if not check_implies(sys_a, sys_b, 'A', 'B', **kwd):
        status |= 1
    if not opts['--one-way']:
        if not check_implies(sys_b, sys_a, 'B', 'A', **kwd):
            status |= 2
    return status


if __name__ == '__main__':
    sys.exit(main())
