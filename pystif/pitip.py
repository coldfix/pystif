"""
Python Information Theoretic Inequality Prover

Usage:
    pitip VARNAMES CHECK [-p]

Options:
    -p, --prove         Show proof
"""

import numpy as np

from .core.app import application
from .core.io import System, column_varname_labels
from .core.it import elemental_inequalities, elemental_forms


def _format_dual_coef(system, index, coef):
    expr = system.row_names[index]
    if np.isclose(coef, 1):
        return expr
    if np.isclose(coef, round(coef)):
        coef = int(round(coef))
    return "{} * {}".format(expr, coef)


def format_dual_vector(system, vec):
    formatted = (
        _format_dual_coef(system, index, coef)
        for index, coef in enumerate(vec)
        if not np.isclose(coef, 0))
    by_quant_len = lambda s: (-len(s.split()[0]), s)
    return "  " + "\n+ ".join(sorted(formatted, key=by_quant_len))


def get_term_counts(system, vec):
    d = {}
    for coef, cat in zip(vec, system.row_categ):
        if np.isclose(coef, round(coef)):
            coef = int(round(coef))
        if coef != 0:
            d.setdefault(cat, 0)
            d[cat] += coef
    return [d[i] for i in sorted(d, reverse=True)]


def indent(lines, amount, ch=' '):
    padding = amount * ch
    return padding + ('\n'+padding).join(lines.split('\n'))


@application
def main(app):
    opts = app.opts

    varnames = opts['VARNAMES']
    num_vars = len(varnames)
    system = System(np.array(list(elemental_inequalities(num_vars))),
                    column_varname_labels(varnames))

    check = System.load(opts['CHECK'])
    system, _ = system.slice(check.columns, fill=True)
    system.row_names = [f.fmt(varnames) for f in elemental_forms(num_vars)]
    system.row_categ = [len(n)/2-1 + (n[0]=='H')
                        for n in system.row_names]

    lp = system.lp()

    for ineq in check.matrix:
        valid = lp.implies(ineq, embed=True)
        print(valid)
        if valid:
            print(indent(format_dual_vector(system, lp.get_dual_solution()), 4))
            print(get_term_counts(system, lp.get_dual_solution()))
