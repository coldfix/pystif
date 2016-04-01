"""
Find quantum violations for the tripartite bell scenario.

Usage:
    qviol INPUT [-c CONSTR] [-o FILE] [-n NUM] [-d DIMS]
    qviol summary INPUT

Arguments:
    INPUT                               File with known faces of the local cone

Options:
    -o FILE, --output FILE              Set output file
    -c CONSTR, --constraints CONSTR     Optimization constraints (CHSH|CHSHE|PPT|CGLMP)
    -n NUM, --num-runs NUM              Number of searches for each inequality [default: 10]
    -d DIMS, --dimensions DIMS          Hilbert space dimensions of subsystems [default: 222]
"""

from operator import matmul
from functools import reduce, partial
from itertools import product
from math import log2, sin, cos, pi
import cmath
import sys

import scipy.optimize
import numpy as np
import yaml

from .core.app import application
from .core.symmetry import SymmetryGroup, group_by_symmetry
from .core.util import _varset
from .core.io import (System, yaml_dump, format_human_readable,
                      read_system_from_file)
from .core.linalg import (projector, measurement, random_direction_vector,
                          cartesian_to_spherical, kron, to_unit_vector,
                          to_quantum_state, ptrace, ptranspose, dagger,
                          as_column_vector)


def exp_i(phi):
    """Return the complex unit vector ``exp(iφ)``."""
    return cmath.rect(1, phi)


def cos_sin(phi):
    """Return ``(cos(φ), sin(φ))``."""
    z = exp_i(phi)
    return z.real, z.imag


def measure_many(psi, measurements):
    """Return expectation values for a list of operators."""
    return [measurement(psi, m) for m in measurements]


class CompositeQuantumSystem:

    def __init__(self, subdim):
        """
        :param list subdim: list of subsystem dimensions
        """
        self.subdim = subdim
        self.cumdim = [1] + list(np.cumprod(subdim))
        self.dim = self.cumdim[-1]

    def lift(self, operator, subsys):
        """
        Lift an operator defined on a subsystem.

        :param np.ndarray operator: matrix for subsystem operator
        :param int subsys: subsystem index
        """
        l = np.eye(self.cumdim[subsys])
        r = np.eye(self.dim / self.cumdim[subsys+1])
        return kron(l, operator, r)

    def lift_all(self, parties):
        """
        Generate measurements from
        """
        return [[tuple(self.lift(mi, i) for mi in m)
                 for m in povms]
                for i, povms in enumerate(parties)]


def umat_V(params, dim):
    """
    Create a unitary matrix using `(n-1)²` real parameters from the given
    parameter list.

    Any unitary matrix `X` can be expressed as a product of three unitaries:

        X = L V R

    where L and R are diagonal matrices responsible for phase transformations.
    This function returns the non-diagonal matrix V in the middle.

    This is an implementation of the parametrization described in:

        C. Jarlskog. A recursive parametrization of unitary matrices. Journal
        of Mathematical Physics, 46(10), 2005.
    """
    if dim == 1:
        return np.eye(1)
    elif dim == 2:
        a = [1]
    elif dim == 3:
        exp_gamma = exp_i(params.pop())
        exp_delta = exp_i(params.pop())
        a = [exp_gamma.real,
             exp_gamma.imag * exp_delta]
    else:
        raise NotImplementedError("Currently only implemented dim <= 3")

    a = as_column_vector(a)
    a_ = dagger(a)

    c, s = cos_sin(params.pop())

    n = dim - 1
    I = np.eye(n)
    L = np.array(np.bmat([
        [I - (1 - c) * (a@a_),  s * a],
        [-s * a_,               [[c]]],
    ]))
    R = np.array(np.bmat([
        [umat_V(params, n),     np.zeros((n, 1))],
        [np.zeros((1, n)),      np.ones((1, 1))],
    ]))
    return L @ R


def unpack_unitary(params, dim):
    """
    Create a `dim` dimensional unitary matrix using (popping) `dim²` real
    parameters from the given parameter list.
    """
    phases = [exp_i(params.pop()) for _ in range(2*dim-1)]
    Phi_L = np.diag(phases[:dim])
    Phi_R = np.diag(phases[dim:]+[1])
    V = umat_V(params, dim)
    return Phi_L @ V @ Phi_R


class TripartiteBellScenario(CompositeQuantumSystem):

    symm = 'Aa <> aA; AaBb <> BbAa; AaCc <> CcAa'

    def __init__(self, system, dims=(2, 2, 2)):
        super().__init__(dims)
        self.cols = system.columns
        sg = SymmetryGroup.load(self.symm, self.cols)
        self.rows = [g[0] for g in group_by_symmetry(sg, system.matrix)]
        # If 'mixing' is enabled the parameter list is extended by an
        # additional mixing probability with the unit matrix. This induces
        # states to be returned as density matrices rather than vectors:
        self.mixing = False

    def random(self):
        # phases are absorbed into Phi_R:
        s = random_direction_vector(self.dim*2)
        # 1 measurement = U(n).CT @ diag[1 … n] @ U(n)
        num_unitary_params = (sum(x**2 for x in self.subdim) +
                              sum(x**2 for x in self.subdim[1:]))
        u = np.random.uniform(0, 2*pi, size=num_unitary_params)
        if self.mixing:
            return np.hstack((s, u, 0))
        return np.hstack((s, u))

    def unpack(self, params):
        l = list(params)

        if self.mixing:
            p_mix = l.pop()

        D = np.diag(range(self.dim))
        U = [[np.eye(self.subdim[0], dtype=complex),
              unpack_unitary(l, self.subdim[0])]]
        U += [[unpack_unitary(l, subdim) for _ in range(2)]
              for subdim in self.subdim[1:]]
        #M = [dagger(u) @ D @ u for u in U]
        #M = list(zip(M[::2], M[1::2]))

        state = to_quantum_state(to_unit_vector(l).reshape(self.dim, 2))

        if self.mixing:
            state = ((1-p_mix) * projector(state) +
                     p_mix * np.eye(self.dim) / self.dim)

        return state, U

    def realize(self, params):
        state, bases = self.unpack(params)
        projectors = [[[projector(u) for u in basis.T]
                       for basis in party]
                      for party in bases]
        return state, projectors

    def _mkcombo(self, parties, cols):
        # measurement combinations
        # initial:  [[(y, n) per angle] per party]        in small hilbert spaces
        # lift:     [[(y, n) per angle] per party]        in large hilbert space
        # select:   [[(ay, an), (by, bn)] per operator pair among different parties]
        # product:  [[(ay,by), (ay,bn), (an,by), (an,bn)] per operator pair]
        # matmul:   [[ay@by, ay@bn, an@by, an@bn] per operator pairing (a,b)]
        _ = self.lift_all(parties)
        _ = select_combinations(_, cols)
        _ = [product(*m) for m in _]
        _ = [[reduce(matmul, parts) for parts in m] for m in _]
        return _

    def violation(self, params, expr):
        state, parties = self.realize(params)
        mcombo = self._mkcombo(parties, self.cols)
        measured = [measure_many(state, m) for m in mcombo]
        entropies = [H(x) for x in measured]
        return np.dot(entropies, expr)


class Constraints:

    """Base class for optimization constraints."""

    def __init__(self, system):
        self.system = system

    def __call__(self, params):
        """Compute constraint functions from packed parameter list."""
        state, parties = self.system.realize(params)
        return self.evaluate_all_constraints(state, parties)


class LinearConstraints(Constraints):

    """Base class for constraints that are linear expressions on some vector."""

    def __init__(self, system):
        self.system = system
        self.matrix = np.array(list(self.get_matrix()))

    def evaluate_all_constraints(self, state, parties):
        vector = self.get_data_vector(state, parties)
        values = self.matrix @ vector
        return self.condition(values)

    def get_matrix(self):
        sg = self.get_symmetry_group()
        expr = self.expr
        if isinstance(expr, dict):
            expr = np.array([expr.get(c, 0) for c in self.cols])
        return sg(expr)

    def get_symmetry_group(self):
        return SymmetryGroup.load(self.system.symm, self.cols)


class CHSHE2(LinearConstraints):

    """H(A,B) + H(A,b) + H(a,B) - H(a,b) - H(A) - H(B) >= 0"""

    cols = ('_AB', '_Ab', '_aB', '_ab',
            '_AC', '_Ac', '_aC', '_ac',
            '_BC', '_Bc', '_bC', '_bc',
            '_A', '_a', '_B', '_b', '_C', '_c')
    expr = {'_AB': 1, '_Ab': 1, '_aB': 1,
            '_ab': -1, '_A': -1, '_B': -1}

    def get_data_vector(self, state, parties):
        """Return the entropies `H(X,Y)`, `H(X)`."""
        mcombo = self.system._mkcombo(parties, self.cols)
        return [H(measure_many(state, m)) for m in mcombo]

    def condition(self, x):
        """Return a negative value if `x >= 0` is not satisfied."""
        return x


class CHSH2(LinearConstraints):

    """E(A,B) + E(A,b) + E(a,B) - E(a,b) <= 2"""

    cols = ('_AB', '_Ab', '_aB', '_ab',
            '_AC', '_Ac', '_aC', '_ac',
            '_BC', '_Bc', '_bC', '_bc',)
    expr = {'_AB': 1, '_Ab': 1, '_aB': 1, '_ab': -1}

    def get_data_vector(self, state, parties):
        """Return the correlators `E(X,Y) = <XY>`."""
        mcombo = select_combinations(self.system.lift_all(parties), self.cols)
        mcombo = [(a[0]-a[1]) @ (b[0]-b[1]) for a, b in mcombo]
        return measure_many(state, mcombo)

    def condition(self, x):
        """Return negative value if `|x| <= 2` is not satisfied."""
        return 2 - abs(x)


class CGLMP2(LinearConstraints):

    """
    Evaluate the CGLMP constraint `I_d ≤ 2` for d outcomes.

    The constraint is defined by:

        I_d = sum_{k=0}^{[d/2]-1} (
                (P(A1=B1+k) + P(B1=A2+1+k) + P(A2=B2+k) + P(B2=A1+k))
                - (P(A1=B1-1-k) + P(B1=A2-k) + P(A2=B2-1-k) + P(B2=A1-1-k))
            )
    """

    cols = ('_AB', '_Ab', '_aB', '_ab',
            '_AC', '_Ac', '_aC', '_ac',
            '_BC', '_Bc', '_bC', '_bc',)

    def __init__(self, system, d=3):
        self.d = d
        super().__init__(system)

    @property
    def expr(self):
        d = self.d
        # Get the expression for AaBb
        AB, Ab, aB, ab = 0, 1, 2, 3
        expr = np.zeros((12, d, d))
        for k in range(d//2):
            # multiplied by (d-1) to avoid floats:
            v = 1 - 2*k/(d-1)
            for i in range(d):
                # positive terms
                expr[AB,i,(i-k)%d] += v     # P(A1=B1+k)    = P(B1=A1-k)
                expr[aB,i,(i+1+k)%d] += v   # P(B1=A2+k+1)  = P(B1=A2+k+1)
                expr[ab,i,(i-k)%d] += v     # P(A2=B2+k)    = P(B2=A2-k)
                expr[Ab,i,(i+k)%d] += v     # P(B2=A1+k)    = P(B2=A1+k)
                # negative terms
                expr[AB,i,(i+k+1)%d] -= v   # P(A1=B1-k-1)  = P(B1=A1+k+1)
                expr[aB,i,(i-k)%d] -= v     # P(B1=A2-k)    = P(B1=A2-k)
                expr[ab,i,(i+k+1)%d] -= v   # P(A2=B2-k-1)  = P(B2=A2+k+1)
                expr[Ab,i,(i-k-1)%d] -= v   # P(B2=A1-k-1)  = P(B2=A1-k-1)
        return expr.flatten()

    def get_symmetry_group(self):
        d = self.d
        var_names = np.arange(3*2*d).reshape((3*2, d))
        var_names = [list(n) for n in var_names]
        col_names = [frozenset(var_names["AaBbCc".index(v)][k]
                               for v, k in zip(c[1:], ab))
                     for c in self.cols
                     for ab in product(range(d), range(d))]
        spec = [(var_names[0]+var_names[1], var_names[1]+var_names[0]),
                (var_names[0]+var_names[1], var_names[2]+var_names[3]),
                (var_names[0]+var_names[1], var_names[4]+var_names[5])]
        return SymmetryGroup.load(spec, col_names)

    def get_data_vector(self, state, parties):
        """
        Return probabilities for each combination of outcomes of
        simultaneously allowed measurements.
        """
        mcombo = select_combinations(self.system.lift_all(parties), self.cols)
        mcombo = [x @ y
                  for a, b in mcombo
                  for x, y in product(a, b)]
        return measure_many(state, mcombo)

    def condition(self, x):
        """Return negative value if `|x| <= 2` is not satisfied."""
        return 2 - abs(x)


class PPT2(Constraints):

    """The 2-party subsystems are separable."""
    # Peres–Horodecki criterion

    # NOTE: this test only works if at most one party is 3 dimensional

    def __init__(self, system):
        self.system = system

    def __call__(self, params):
        """Compute constraint functions from packed parameter list."""
        state, parties = self.system.realize(params)
        constr = self.evaluate_all_constraints(state, parties)
        neg = sum(x for x in constr if x < 0)
        pos = sum(x for x in constr if x > 0)
        return [neg if neg < 0 else pos]

    def evaluate_all_constraints(self, state, parties):
        rho_abc = projector(state)
        dims = self.system.subdim
        rho_bc = ptrace(rho_abc, dims, 0), (dims[1], dims[2])
        rho_ac = ptrace(rho_abc, dims, 1), (dims[0], dims[2])
        rho_ab = ptrace(rho_abc, dims, 2), (dims[0], dims[1])
        vals = [self._neg_entanglement(rho2, dim2)
                for rho2, dim2 in (rho_bc, rho_ac, rho_ab)]
        return np.array(vals).flatten()

    def _neg_entanglement(self, rho2, dim2):
        """
        Return something negative if the 2-party density matrix is entangled.

        This works using the Peres–Horodecki criterion.
        """
        trans = ptranspose(rho2, dim2, 1)
        val, vec = np.linalg.eigh(trans)
        return val


# composed operations
def h(p):
    """Compute one term in the entropy sum."""
    return -p*log2(p) if p > 0 else 0

def H(pdist):
    """Compute entropy from a numpy array of probabilities."""
    return sum(map(h, pdist))


def select_combinations(parties, columns):
    _sel_op = lambda p: parties[ord(p.lower())-ord('a')][p.islower()]
    return [[_sel_op(p) for p in _varset(colname)]
            for colname in columns]


def load_summary(filename):
    with open(filename) as f:
        data = yaml.safe_load(f)
    results = data['results'] or ()
    indices = sorted({int(r['i_row']) for r in results})
    rows = [data['rows'][i] for i in indices]
    cols = data['cols']
    return indices, rows, cols


def show_summary(opts):
    indices, rows, cols = load_summary(opts['INPUT'])
    print(len(rows), "rows:", ",".join(map(str, indices)))
    for r in rows:
        print(format_human_readable(r, cols))


def get_constraints_obj(constraints_name, system):
    constraint_classes = {
        'CHSHE': CHSHE2,
        'CHSH': CHSH2,
        'PPT': PPT2,
        'CGLMP': CGLMP2,
    }
    if constraints_name is None:
        return []
    cls = constraint_classes[constraints_name.upper()]
    return cls(system)


@application
def main(app):

    opts = app.opts
    dims = list(map(int, opts['--dimensions']))
    if opts['summary']:
        show_summary(opts)
        return

    if opts['INPUT'].lower().endswith('.yml'):
        indices, rows, cols = load_summary(opts['INPUT'])
        app.system = System(np.array(rows), cols)

    system = TripartiteBellScenario(app.system, dims=dims)
    constr = get_constraints_obj(opts['--constraints'], system)

    num_runs = int(opts['--num-runs'])

    if opts['--output']:
        out_file = open(opts['--output'], 'wt')
    else:
        out_file = sys.stdout

    yaml_dump({
        'rows': system.rows,
        'cols': system.cols,
        'expr': [
            format_human_readable(expr, system.cols)
            for expr in system.rows
        ],
        'constr': opts['--constraints'],
    }, out_file)
    print("results:", file=out_file, flush=True)

    for _, (i, expr) in product(range(num_runs), enumerate(system.rows)):

        system.mixing = False
        result = scipy.optimize.minimize(
            system.violation, system.random(),
            (expr,),
            constraints=constr and [
                {'type': 'ineq', 'fun': constr},
            ])

        objective = result.fun
        params = result.x
        state, parties = system.realize(params)
        fconstr = constr and constr.evaluate_all_constraints(state, parties)
        success = False
        reoptimize = False

        i = str(i).ljust(2)

        if not result.success:
            print(i, 'error', result.message)
        elif objective > -1e-11:
            print(i, 'no violation', objective, fconstr)
        elif any(x < 0 for x in fconstr):
            print(i, 'unsatisfied constraint', objective, fconstr)
            reoptimize = objective / min(fconstr) > 100
        else:
            success = True

        if reoptimize:
            print(i, 'reoptimization...')

            def constraint_violation(params):
                return -sum(constr(params))

            def retain_objective(params):
                return -(system.violation(params, expr) - 0.5 * objective)

            system.mixing = True
            result = scipy.optimize.minimize(
                constraint_violation, np.hstack((params, 0.001)),
                constraints=[
                    {'type': 'ineq', 'fun': retain_objective},
                    # Constraint the mixing probability 0 ≤ p ≤ 1:
                    {'type': 'ineq', 'fun': lambda params: params[-1]},
                    {'type': 'ineq', 'fun': lambda params: 1-params[-1]},
                ])

            params = result.x
            objective = system.violation(params, expr)
            state, parties = system.realize(params)
            fconstr = constr.evaluate_all_constraints(state, parties)

            if not result.success:
                print(i, '  -> error', result.message)
            elif any(x < 0 for x in fconstr):
                print(i, '  -> still unsatisfied', objective, fconstr)
            else:
                success = True

        if success:
            print(i, 'success', objective, fconstr)

            state, bases = system.unpack(params)
            yaml_dump([{
                'i_row': int(i),
                'f_objective': objective,
                'f_constraints': fconstr,
                'opt_params': params,
                'state': state,
                'bases': bases,
                'reoptimize': reoptimize,
            }], out_file)
            out_file.flush()

        sys.stdout.flush()
