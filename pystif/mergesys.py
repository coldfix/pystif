"""
Merge systems of linear equations.

Usage:
    mergesys INPUT... [-o OUTPUT]

Options:
    -o OUTPUT, --output OUTPUT          Set output file
"""

import sys
import numpy as np
from docopt import docopt
from .util import System, SystemFile


# TODO: minify while merging?


def main(args=None):
    opts = docopt(__doc__, args)

    system = System()
    for file in opts['INPUT']:
        system = system.merge(System.load(file))

    out = SystemFile(opts['--output'], columns=system.columns)
    for ineq in system.matrix:
        out(ineq)


if __name__ == '__main__':
    main()
