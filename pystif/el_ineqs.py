"""
Output elemental inequalities for given number of variables.

Usage:
    el_ineqs NUM_VARS [-o OUTPUT] [-a]

Options:
    -o OUTPUT, --output OUTPUT      Write inequalities to this file
    -a, --append                    Append to output file

This will output a matrix of inequalities for NUM_VARS random variables. Each
row ``q`` corresponds to one inequality

    q∙x ≥ 0.

The columns correspond to the entropies of

    X₀, X₁, X₀X₁, X₂, X₀X₂, X₁X₂, X₀X₁X₂, …

and so on, i.e. the bit representation of the column number corresponds to the
subset of variables.
"""

import sys
from docopt import docopt
from .core.it import elemental_inequalities
from .util import format_vector, print_to


def main(args=None):
    opts = docopt(__doc__, args)
    print_ = print_to(opts['--output'], opts['--append'])

    num_vars = int(opts['NUM_VARS'])

    for v in elemental_inequalities(num_vars):
        print_(format_vector(v[1:]))


if __name__ == '__main__':
    main()
