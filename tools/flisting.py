"""
Facet listing

Usage:
    flisting.py INPUT [-o PREFIX] [-w PREFIX] [-m MODE]

Options:
    -o PREFIX, --output PREFIX          Prefix for output files
    -w PREFIX, --witnesses PREFIX       Prefix for witness files (qviol)
    -m MODE, --mode MODE                System type [default: bell]
"""

import os
import yaml
from collections import OrderedDict, defaultdict
from functools import reduce
from itertools import groupby
from operator import itemgetter as get

from docopt import docopt

from pystif.core import it
from pystif.core.io import (System, _sort_col_indices, _coef,
                            format_vector, column_varname_labels)
from pystif.core.symmetry import group_by_symmetry
from pystif.core.util import scale_to_int, _varset, varsort
from pystif.core.lp import Problem


def int_vector(v):
    return tuple(int(round(x)) for x in v)


def grouped(l, key):
    return [list(g) for k, g in groupby(sorted(l, key=key), key=key)]


def load_witnesses(filename, cols, ineqs):
    with open(filename) as f:
        data = yaml.safe_load(f)
    results = data['results'] or ()

    # establish expected column order:
    col_trans = [data['cols'].index(c) for c in cols]
    file_rows = [int_vector(r[i] for i in col_trans)
                 for r in data['rows']]

    # find index of inequality within global list
    ineq_indices = {v: i for i, v in enumerate(ineqs)}
    file_row_idx = [ineq_indices[r] for r in file_rows]
    for r in results:
        r['i_row'] = file_row_idx[int(r['i_row'])]

    # sort by ineq-index and select only the entry with the maximum violation
    return [
        min(g, key=get('f_objective'))
        for g in grouped(results, get('i_row'))
    ]


def facet_weight(f, cols):
    """A key for sorting facets."""
    w = defaultdict(lambda: (0, 0))
    for coef, col in zip(f, cols):
        num = len(_varset(col))
        p, n = w[num]
        if coef > 0:
            p += coef
        elif coef < 0:
            n += -coef
        w[num] = p, n
    weights = [w[i] for i in sorted(w, reverse=True)]
    return [*weights, f]


def raw_breqn(eqns):
    yield r'\begin{dgroup*}'
    for lhs, rhs in eqns:
        yield r'\begin{dmath*}' #+ r'\breakingcomma'
        yield r'  {} = {}'.format(lhs, rhs)
        yield r'\end{dmath*}'
    yield '\\end{dgroup*}'


def breqn(eqns):
    yield from raw_breqn((lhs, ' '.join(rhs).lstrip('+ '))
                         for lhs, rhs in eqns)


def align(eqns):
    yield r'\begin{align*}'
    yield '\\\\[1mm]\n'.join(
        lhs.ljust(6) + ' &= ' + ' '*14 + split_expr(rhs)
        for lhs, rhs in eqns)
    yield r'\end{align*}'


def split_expr(terms, max_width=55):
    rlines = ['']
    est_w = -2
    for term in terms:
        w = estimate_expr_width(term)
        if est_w + w >= max_width:
            rlines.append(term)
            est_w = 1+w
        else:
            rlines[-1] += term
            est_w += w
    rlines[0] = rlines[0].lstrip('+ ')
    return "\\\\*\n    &\phantom{=}\quad   ".join(rlines)


def estimate_expr_width(expr):
    i = 0
    w = 0
    while i < len(expr):
        c = expr[i]
        if c in '+-':
            w += 2
        elif c == '_':
            i += 1
            if expr[i] == '{':
                i += 1
                while expr[i] != '}':
                    w += 0.5
                    i += 1
            else:
                w += 0.5
        elif c != ' ':
            w += 1
        i +=  1
    return w


def format_latex(ineqs, cols, mode):


    if mode == 'bell':
        varname_translate = {
            'A': '\\A_1',
            'a': '\\A_2',
            'B': '\\B_1',
            'b': '\\B_2',
            'C': '\\C_1',
            'c': '\\C_2',
        }
        fmt = "H({})"
        sep = "\\,"
    else:
        varnames = varsort({v for c in cols for v in _varset(c)})
        varname_translate = {v: str(i+1) for i,v in enumerate(varnames)}
        fmt = "H_{{{}}}"
        sep = ""

    cols = [
        fmt.format(sep.join(
            sorted(varname_translate[v] for v in _varset(c))
        ))
        for c in cols
    ]

    lhs = lambda i: r'I_{%i}' % i
    rhs = lambda v: ["{} {}".format(_coef(v[i]), cols[i])
                     for i in _sort_col_indices(v, cols)
                     if cols[i] != '_']

    eqnarray = align
    eqnarray = breqn

    return eqnarray([
        (lhs(i), rhs(ineq))
        for i, ineq in enumerate(ineqs)
    ])


def format_short(ineqs, cols):

    varnames = varsort({v for c in cols for v in _varset(c)})
    col_code = [
        sum(1<<varnames.index(v) for v in _varset(c))
        for c in cols
    ]
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    for ineq in ineqs:
        indices = [i for i, c in enumerate(ineq)
                   if c != 0]
        indices = sorted(indices, key=lambda i: col_code[i])
        rhs = ["{} {}".format(_coef(ineq[i]), alphabet[col_code[i]])
            for i in indices if cols[i] != '_']
        yield " ".join(rhs).replace(" ", "")


def format_short_2(ineqs, cols):
    cols = ["".join(varsort(_varset(c))) for c in cols]
    for ineq in ineqs:
        indices = _sort_col_indices(ineq, cols)
        rhs = ["{} {}".format(_coef(ineq[i]), cols[i])
            for i in indices if cols[i] != '_']
        yield " ".join(rhs).replace(" ", "").lstrip('+')


def format_short_3(ineqs, cols, mode):
    if mode == 'bell':
        return format_short_bell(ineqs, cols)
    else:
        return format_short_cca(ineqs, cols)


def format_short_bell(ineqs, cols):
    yield """
# Facet description of the {:02}D local cone in a tripartite Bell scenario.
# Each matrix row `R` contains the coefficients of a linear inequality
#
#   R*H >= 0
#
# where `H` is an entropy vector with the following columns in order:
""".strip().format(len(cols))
    yield "#:: " + " ".join(cols)
    yield """
#
# The following substitutions are implied:
#
#~~  A <> a
#~~ Aa <> Bb
#~~ Aa <> Cc
""".lstrip()
    for ineq in ineqs:
        yield format_vector(ineq)


def format_short_cca(ineqs, cols):
    varnames = "".join(varsort({v for c in cols for v in _varset(c)}))
    num_vars = len(varnames)
    yield """
# Facet description of the information cone for the width {} cyclic CCA.
# Each matrix row `R` contains the coefficients of a linear inequality
#
#   R*H >= 0
#
# where `H` is an entropy vector with the following columns in order:
""".strip().format(num_vars)
    yield "#:: " + " ".join(cols)
    yield """
#
# The facets of the Shannon cone are omitted from the list.
# The following substitutions are implied:
#
#~~ {0} <> {1}; {0} <> {2}
""".lstrip().format(varnames, varnames[1:]+varnames[0], varnames[::-1])
    for ineq in ineqs:
        yield format_vector(ineq)


def load_all_witnesses(prefix, cols, ineqs):
    prefix += '{:02}D-'.format(len(cols))

    wit = witnesses = OrderedDict()
    wit[2, 'none' ] = load_witnesses(prefix + "222-none.yml", cols, ineqs)
    if not wit[2, 'none']:
        return {}
    wit[2, 'chshe'] = load_witnesses(prefix + "222-CHSHE.yml", cols, ineqs)
    wit[2, 'chsh' ] = load_witnesses(prefix + "222-CHSH.yml", cols, ineqs)
    wit[2, 'ppt'  ] = load_witnesses(prefix + "222-PPT.yml", cols, ineqs)
    wit[3, 'none' ] = load_witnesses(prefix + "333-none.yml", cols, ineqs)
    wit[3, 'chshe'] = load_witnesses(prefix + "333-CHSHE.yml", cols, ineqs)
    wit[3, 'cglmp'] = load_witnesses(prefix + "333-CGLMP.yml", cols, ineqs)
    wit[3, 'ppt'  ] = load_witnesses(prefix + "333-PPT.yml", cols, ineqs)
    return wit


def witness_info(witnesses):
    for _, w in witnesses.items():
        yield " ".join(str(r['i_row']) for r in w)


#----------------------------------------
# STUFF FOR STATE LISTING
#----------------------------------------

from pystif import qviol
from pystif.qviol import exp_i, cos_sin
from math import cos, sin
from numpy.testing import assert_allclose
import numpy as np

def simplify_params(params, dim):
    phases = [params.pop() for _ in range(2*dim-1)]
    alpha = phases[:dim]
    beta  = phases[dim:]
    if dim == 2:
        gamma = [params.pop()]
    else:
        gamma = [params.pop() for _ in range(4)]
    return alpha, beta, gamma

def implement_simple_3(alpha, beta, gamma):
    Phi_L = np.diag([exp_i(a) for a in alpha])
    Phi_R = np.diag([exp_i(b) for b in beta]+[1])
    a, b, c, d = gamma
    #----------------------------------------
    # V_L = np.array([
    #     [1-(1-cos(c))*cos(a)**2,
    #         (1-cos(c))*cos(a)*sin(a)*exp_i(-b),
    #         sin(c)*cos(a)],
    #     [(1-cos(c))*cos(a)*sin(a)*exp_i(b),
    #         1-(1-cos(c))*sin(a)**2,
    #         sin(c)*sin(a)*exp_i(b)],
    #     [-sin(c)*cos(a),
    #         -sin(c)*sin(a)*exp_i(b),
    #         cos(c)],
    # ])
    #----------------------------------------

    exp_gamma = exp_i(a)
    exp_delta = exp_i(b)
    a = [exp_gamma.real,
         exp_gamma.imag * exp_delta]

    a = qviol.as_column_vector(a)
    a_ = qviol.dagger(a)

    c, s = cos_sin(c)
    I = np.eye(2)
    V_L = np.array(np.bmat([
        [I - (1 - c) * (a@a_),  s * a],
        [-s * a_,               [[c]]],
    ]))

    V_R = np.array([
        [ cos(d),   sin(d), 0],
        [-sin(d),   cos(d), 0],
        [      0,        0, 1],
    ])
    return Phi_L @ V_L @ V_R @ Phi_R

def implement_simple_2(alpha, beta, gamma):
    Phi_L = np.diag([exp_i(a) for a in alpha])
    Phi_R = np.diag([exp_i(b) for b in beta]+[1])
    a, = gamma
    V = np.array([
        [cos(a), sin(a)],
        [-sin(a), cos(a)],
    ])
    return Phi_L @ V @ Phi_R

def explicit_params(params, dim):
    params = list(params)
    for k in range(5):
        p1 = params
        p2 = params[:]
        m1 = qviol.unpack_unitary(p1, dim)
        alpha, beta, gamma = simplify_params(p2, dim)
        # early consistency check
        if dim == 2:
            m2 = implement_simple_2(alpha, beta, gamma)
        else:
            m2 = implement_simple_3(alpha, beta, gamma)
        assert len(p1) == len(p2)
        assert_allclose(m1, m2)
        yield alpha, beta, gamma


trumped_by = {
    (2, 'none'): [(2, 'chshe')],
    (2, 'chshe'): [(2, 'chsh')],
    (2, 'chsh'): [(2, 'ppt')],
    (2, 'ppt'): [],
    (3, 'none'): [(3, 'cglmp'), (2, 'none')],
    (3, 'cglmp'): [(3, 'chshe')],
    (3, 'chshe'): [(3, 'ppt'), (2, 'chshe')],
    (3, 'ppt'): [(2, 'ppt')],
}


def select_states(wit):
    """Select strongest claim for each inequality."""
    indices_all = {
        k: {r['i_row'] for r in w}
        for k, w in wit.items()
    }
    is_trumped = lambda k, i: any(i in indices_all[p] for p in trumped_by[k])
    indices_sliced = {
        k: {i for i in w if not is_trumped(k, i)}
        for k, w in indices_all.items()
    }
    return OrderedDict([
        (k, [r for r in w if r['i_row'] in indices_sliced[k]])
        for k, w in wit.items()
    ])


def _fmt_num(n):
    return r'{{%s}%.6f}' % ('+' if n >= 0 else '-', abs(n))

def _format_cmplx(z):
    r, i = z
    return r'{%s\,%si}' % (_fmt_num(r), _fmt_num(i))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

def _fmt_vec(v):
    return r'\smash{\big(}&&%s\,\smash{\big)}' % (
        ',\\\\\n& && '.join(
            ',\\,'.join(map(_fmt_num, chunk))
            for chunk in chunks(v, 6)))

def _format_state(k, r):
    v = r['state']
    lhs = r'\vec\psi'.format(r['i_row'], k[0])
    rhs = r'\smash{\big(}&&%s\,\smash{\big)}' % (
        ',\\\\\n& && '.join(
            ',\\,'.join(map(_format_cmplx, chunk))
            for chunk in chunks(v, 3)))
    return lhs, rhs

def _format_bases(k, r):
    if len(r['state']) == 8:
        dim = 2
    elif len(r['state']) == 27:
        dim = 3
    params = list(r['opt_params'])
    if r.get('reoptimize'):
        params.pop()
    names = [
        r'\vec\alpha_2',
        r'\vec\beta_1',
        r'\vec\beta_2',
        r'\vec\gamma_1',
        r'\vec\gamma_2',
    ]
    for n, (a, b, c) in zip(names, explicit_params(params, dim)):
        yield (n, _fmt_vec(a + b + c))


def _format_mixing(r):
    if r.get('reoptimize'):
        mixing = '&&{:.6f}'.format(r.get('mixing'))
        yield (r'\lambda', mixing)

def _alignat(eqns):
    yield r'\begin{alignat*}{2}'
    yield '\\\\\n'.join(
        r'  {} &= {}'.format(lhs, rhs)
        for lhs, rhs in eqns
    )
    yield r'\end{alignat*}'


def state_listing(flat):
    for k, r in flat:
        if k[1] == 'none':
            remark = ''
        else:
            remark = ' satisfying bipartite {}'.format(k[1].upper())
        yield r'\noindent'
        #yield r'Violation:'
        yield (r'$I_{%s}\cdot\vec h_{\vec\psi} = %.5f$' % (r['i_row'], r['f_objective'])
               +r' for $\vec\psi\in\Qunits{%d}$%s:' % (k[0], remark))
        yield from _alignat([
            *_format_mixing(r),
            _format_state(k, r),
            *_format_bases(k, r)
        ])
        yield r''


def make_state_listing(wit, basename):

    flat = sorted(
        ((k, r) for k, w in wit.items() for r in w),
        key=lambda r: r[1]['i_row'])

    state_all = "\n".join(state_listing(flat))

    with open(basename + '.state.tex', 'wt') as f:
        f.write(state_all)

#----------------------------------------
# OTHER STUFF AGAIN
#----------------------------------------


def read_bell_file(filename):
    system = System.load(filename)
    cols = sorted(system.columns, key=lambda c: (len(c), c.lower(), c))
    system, _ = system.slice(cols)

    groups = group_by_symmetry(system.symmetry_group(), system.matrix)
    ineqs = [int_vector(g[0]) for g in groups]
    ineqs = sorted(ineqs, key=lambda f: facet_weight(f, cols))

    return ineqs, cols


def read_cca_file(filename):
    system = System.load(filename)
    varnames = varsort({v for c in system.columns for v in _varset(c)})
    num_vars = len(varnames)
    colnames = column_varname_labels(varnames)[1:]
    system, _ = system.slice(colnames)

    groups = group_by_symmetry(system.symmetry_group(), system.matrix)
    ineqs = [int_vector(g[0]) for g in groups]
    ineqs = sorted(ineqs, key=lambda f: facet_weight(f, colnames))

    basic = Problem([f[1:] for f in it.elemental_inequalities(num_vars)])
    ineqs = [ineq for ineq in ineqs if not basic.implies(ineq)]

    return ineqs, colnames



def main(args=None):
    opts = docopt(__doc__, args)

    oprefix = opts['--output'] or ''
    wprefix = opts['--witnesses'] or ''
    mode = opts['--mode']

    print("  - loading system...")
    if mode == 'bell':
        ineqs, cols = read_bell_file(opts['INPUT'])
        dim = len(cols)
        basename = '{}bell-{:02}D'.format(oprefix, dim)
    else:
        ineqs, cols = read_cca_file(opts['INPUT'])
        num_vars = it.num_vars(len(cols)+1)
        basename = '{}cca-{}'.format(oprefix, num_vars)

    make_latex = True
    make_compact = True
    make_witness = bool(wprefix)

    if make_latex:
        print("  - formatting equations...")
        with open(basename + '.tex', 'wt') as f:
            f.write("\n".join(format_latex(ineqs, cols, mode)))

    # print("\n".join(format_short_2(ineqs, cols)))

    if make_compact:
        with open(basename + '.txt', 'wt') as f:
            f.write("\n".join(format_short_3(ineqs, cols, mode)))

    if make_witness:
        print("  - loading witnesses...")
        wit = load_all_witnesses(wprefix, cols, ineqs)
        with open(basename + '.wit', 'wt') as f:
            f.write("\n".join(witness_info(wit)))

        # select subset of states that support the strongest claim:
        print("  - formatting states...")
        wit = select_states(wit)
        make_state_listing(wit, basename)


if __name__ == '__main__':
    main()
