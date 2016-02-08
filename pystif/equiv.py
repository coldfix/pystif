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

from docopt import docopt

from .core.io import format_vector, System
from .core.symmetry import group_by_symmetry


def check_implies(sys_a: System, sys_b: System,
                  name_a: str, name_b: str,
                  *, symmetries: 'SymmetryGroup', quiet=0):
    """
    Check if A implies B (system of linear inequalities).

    The amount of output is controlled by the value of ``quiet``:

        quiet=0     Full output, including the list of missing constraints
        quiet=1     Short output, no list of constraints
        quiet=2     No output at all
    """
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

    sys_a.update_symmetries(opts['--symmetry'])

    status = 0
    kwd = {'quiet': opts['--quiet'], 'symmetries': sys_a.symmetry_group()}
    if not check_implies(sys_a, sys_b, 'A', 'B', **kwd):
        status |= 1
    if not opts['--one-way']:
        if not check_implies(sys_b, sys_a, 'B', 'A', **kwd):
            status |= 2
    return status


if __name__ == '__main__':
    sys.exit(main())
