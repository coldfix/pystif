"""
Find quantum violations for the tripartite bell scenario.

Usage:
    qviol INPUT [-c CONSTR] [-o FILE] [-n NUM] [-s SUBSET] [-d DIMS]

Arguments:
    INPUT                               File with known faces of the local cone

Options:
    -o FILE, --output FILE              Set output file
    -c CONSTR, --constraints CONSTR     Optimization constraints (CHSH|CHSHE|SEP|CGLMP)
    -n NUM, --num-runs NUM              Number of searches for each inequality [default: 10]
    -s SUBSET, --select SUBSET          Select subset of inequalities
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
from .core.io import (System, _varset, yaml_dump, format_human_readable,
                      read_system_from_file)
from .core.linalg import (projector, measurement, random_direction_vector,
                          cartesian_to_spherical, kron, to_unit_vector,
                          to_quantum_state, ptrace, ptranspose, dagger,
                          as_column_vector)


def exp_i(phi):
    return cmath.rect(1, phi)


def cos_sin(phi):
    z = exp_i(phi)
    return z.real, z.imag


def measure_many(psi, measurements):
    """Return expectation values for a list of operators."""
    return [measurement(psi, m) for m in measurements]


class Qbit:

    dim = 2

    sigma_x = np.array([[0,   1], [ 1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1,   0], [0, -1]])
    sigma = np.array((sigma_x, sigma_y, sigma_z))

    @classmethod
    def rotspin(cls, direction):
        r, theta, phi = cartesian_to_spherical(direction)
        return cls.rotspin_angles(theta, phi)

    @classmethod
    def rotspin_slow(cls, direction):
        """Equivalent to rotspin(), just a bit slower."""
        spinmat = np.tensordot(direction, cls.sigma, 1)
        # eigenvalues/-vectors of hermitian matrix, in ascending order:
        val, vec = np.linalg.eigh(spinmat)
        return (projector(vec[:,[1]]),
                projector(vec[:,[0]]))

    @classmethod
    def rotspin_angles(cls, theta, phi):
        """
        Returns the eigenspace projectors for the spin matrix along an
        arbitrary direction.

        The eigenstates are:

            |+> = [cos(θ/2),
                   sin(θ/2) exp(iφ)]

            |-> = [sin(θ/2),
                   cos(θ/2) exp(-iφ) (-1)]

        http://www.physicspages.com/2013/01/19/spin-12-along-an-arbitrary-direction/
        """
        exp_th2 = exp_i(theta/2)
        exp_phi = exp_i(phi)
        u = [exp_th2.real,
             exp_th2.imag * exp_phi]
        d = [exp_th2.imag,
             exp_th2.real * exp_phi * -1]
        return projector(u), projector(d)

    @classmethod
    def xzspin(cls, angle):
        return cls.rotspin_angles(angle, 0)


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

    def random(self):
        # phases are absorbed into Phi_R:
        s = random_direction_vector(self.dim*2)
        # 1 measurement = U(n).CT @ diag[1 … n] @ U(n)
        num_unitary_params = (sum(x**2 for x in self.subdim) + 
                              sum(x**2 for x in self.subdim[1:]))
        u = np.random.uniform(0, 2*pi, size=num_unitary_params)
        return np.hstack((s, u))

    def unpack(self, params):
        l = list(params)

        D = np.diag(range(self.dim))
        U = [[np.eye(self.subdim[0]),
              unpack_unitary(l, self.subdim[0])]]
        U += [[unpack_unitary(l, subdim) for _ in range(2)]
              for subdim in self.subdim[1:]]
        #M = [dagger(u) @ D @ u for u in U]
        #M = list(zip(M[::2], M[1::2]))

        state = to_quantum_state(to_unit_vector(l).reshape(self.dim, 2))
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

    def __init__(self, system):
        sg = SymmetryGroup.load(system.symm, self.cols)
        self.system = system
        expr = np.array([self.coef.get(c, 0) for c in self.cols])
        self.expressions = list(sg(expr))

    def __call__(self, params):
        state, parties = self.system.realize(params)
        measured = self._measure_all(state, parties)
        vals = (self._eval(expr, measured) for expr in self.expressions)
        return sum(v for v in vals if v < 0)

    @classmethod
    def optimization_constraints(cls, system):
        return {'type': 'ineq',
                'fun': cls(system)}


class CHSHE2(Constraints):

    """H(A,B) + H(A,b) + H(a,B) - H(a,b) - H(A) - H(B) >= 0"""

    cols = ('_AB', '_Ab', '_Ba', '_ab',
            '_AC', '_Ac', '_Ca', '_ac',
            '_BC', '_Bc', '_Cb', '_bc',
            '_A', '_a', '_B', '_b', '_C', '_c')
    coef = {'_AB': 1, '_Ab': 1, '_Ba': 1,
            '_ab': -1, '_A': -1, '_B': -1}

    def _measure_all(self, state, parties):
        mcombo = self.system._mkcombo(parties, self.cols)
        return [measure_many(state, m) for m in mcombo]

    def _eval(self, expr, measured):
        entropies = [H(x) for x in measured]
        return np.dot(entropies, expr)


class CHSH2(Constraints):

    """E(A,B) + E(A,b) + E(a,B) - E(a,b) <= 2"""

    cols = ('_AB', '_Ab', '_Ba', '_ab',
            '_AC', '_Ac', '_Ca', '_ac',
            '_BC', '_Bc', '_Cb', '_bc',)
    coef = {'_AB': 1, '_Ab': 1, '_Ba': 1, '_ab': -1}

    def _measure_all(self, state, parties):
        mcombo = select_combinations(self.system.lift_all(parties), self.cols)
        mcombo = [(a[0]-a[1]) @ (b[0]-b[1]) for a, b in mcombo]
        return measure_many(state, mcombo)

    def _eval(self, expr, correlators):
        return 2-abs(np.dot(correlators, expr))


class CGLMP2(Constraints):

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

    def _gen_template_expression(self, d):
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
        return np.array(expr.flat)

    def __init__(self, system, d=3):
        self.system = system
        self.d = d

        var_names = np.arange(3*2*d).reshape((3*2, d))
        var_names = [list(n) for n in var_names]

        col_names = [frozenset(var_names["AaBbCc".index(v)][k]
                               for v, k in zip(c[1:], ab))
                     for c in self.cols
                     for ab in product(range(d), range(d))]

        spec = [(var_names[0]+var_names[1], var_names[1]+var_names[0]),
                (var_names[0]+var_names[1], var_names[2]+var_names[3]),
                (var_names[0]+var_names[1], var_names[4]+var_names[5])]

        sg = SymmetryGroup.load(spec, col_names)

        self.expressions = list(sg(self._gen_template_expression(d)))

    def _measure_all(self, state, parties):
        mcombo = select_combinations(self.system.lift_all(parties), self.cols)
        mcombo = [x @ y
                  for a, b in mcombo
                  for x, y in product(a, b)]
        return measure_many(state, mcombo)

    def _eval(self, expr, probabilities):
        # Recall that we scaled the constraint by the factor (d-1)
        return 2 - abs(np.dot(probabilities, expr))


class SEP2(Constraints):

    """The 2-party subsystems are separable."""
    # Peres–Horodecki criterion

    # NOTE: this test only works if at most one party is 3 dimensional

    def __init__(self, system):
        self.system = system

    def __call__(self, params):
        state, parties = self.system.realize(params)
        rho_abc = projector(state)
        dims = self.system.subdim
        rho_bc = ptrace(rho_abc, dims, 0), (dims[1], dims[2])
        rho_ac = ptrace(rho_abc, dims, 1), (dims[0], dims[2])
        rho_ab = ptrace(rho_abc, dims, 2), (dims[0], dims[1])
        return sum(self._neg_entanglement(rho2, dim2)
                   for rho2, dim2 in (rho_bc, rho_ac, rho_ab))

    def _neg_entanglement(self, rho2, dim2, eps=1e-10):
        """
        Return something negative if the 2-party density matrix is entangled.

        This works using the Peres–Horodecki criterion.
        """
        trans = ptranspose(rho2, dim2, 1)
        val, vec = np.linalg.eigh(trans)
        return sum(v for v in val if v < -eps)


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


@application
def main(app):

    opts = app.opts
    dims = list(map(int, opts['--dimensions']))

    if opts['INPUT'].lower().endswith('.yml'):
        with open(opts['INPUT']) as f:
            data = yaml.safe_load(f)
        if not data:
            return
        cols = data[0]['cols']
        coef = {tuple(r['coef']) for r in data}
        app.system = System(np.array(list(coef)), cols)

    system = TripartiteBellScenario(app.system, dims=dims)

    constraint_classes = {
        'CHSHE': CHSHE2,
        'CHSH': CHSH2,
        'SEP': SEP2,
        'CGLMP': CGLMP2,
    }

    ct = opts['--constraints']
    if ct is None:
        constr = None
    else:
        cls = constraint_classes[ct.upper()]
        constr = cls.optimization_constraints(system)

    num_runs = int(opts['--num-runs'])

    if opts['--output']:
        out_file = open(opts['--output'], 'wt')
    else:
        out_file = sys.stdout

    if opts['--select']:
        select = [int(x) for x in opts['--select'].split(',')]
    else:
        select = range(len(system.rows))

    for _ in range(num_runs):
        for i, expr in enumerate(system.rows):
            if i not in select:
                continue

            result = scipy.optimize.minimize(
                system.violation, system.random(),
                (expr,), constraints=constr)

            if result.fun > -1e-11:
                print('.' if result.success else 'x', end='', flush=True)
                continue

            state, bases = system.unpack(result.x)
            yaml_dump([{
                'i': i,
                'coef': expr,
                'cols': system.cols,
                'expr': format_human_readable(expr, system.cols),
                'f': result.fun,
                'state': state,
                'bases': bases,
            }], out_file)

            print('\n', i, result.fun)

    print()
