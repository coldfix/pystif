"""
Perform a simple Fourier-Motzkin-Elimination (FME) on the rows.

Usage:
    fme INPUT [-o OUTPUT] -s SUBSPACE [-q] [-p] [-d DROP] [-i FILE]

Options:
    -o OUTPUT, --output OUTPUT      Save facets of projected cone
    -s SUB, --subspace SUB          Subspace specification (dimension or file)
    -q, --quiet                     No status output
    -p, --pretty                    Pretty print output inequalities
    -d DROP, --drop DROP            Randomly drop DROP inequalities from the
                                    initial system before beginning the FME
                                    [default: 0]
    -i FILE, --info FILE            Print short summary to file (YAML)
"""

import random

import numpy as np

from .core.app import application
from .core.fme import FME
from .core.io import StatusInfo


class FMEStatusInfo:

    def __init__(self):
        self.info = StatusInfo()
        super().__init__()

    def cb_start(self, rows, cols_to_eliminate):
        num_cols = len(rows[0])
        self.info("Eliminate: {} -> {} columns ({} rows)\n"
                  .format(num_cols, num_cols - len(cols_to_eliminate),
                          len(rows)))

    def cb_step(self, rows, col, z, p, n):
        z, p, n = len(z), len(p), len(n)
        num_cols = len(rows[0])
        self.info("  {:3}, z={:4}, p+n={:3}, p*n={:4}"
                  .format(num_cols, z, p+n, p*n))

    def cb_stop(self, rows):
        self.info("")


class VerboseFME(FMEStatusInfo, FME):
    pass


@application
def main(app):
    fme = FME() if app.quiet else VerboseFME()

    random.seed()
    num_drop = int(app.opts['--drop'])
    for i in range(num_drop):
        d = random.randrange(app.system.shape[0])
        app.system.matrix = np.delete(app.system.matrix, d, axis=0)

    system = app.system
    app.start_timer()
    for row in fme.solve_to(system.matrix, app.subdim):
        app.output(row)
