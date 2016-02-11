"""
Find quantum violations for the tripartite bell scenario.

Usage:
    qviol INPUT [-c CONSTR] [-o FILE] [-n NUM] [-s SUBSET]

Arguments:
    INPUT                               File with known faces of the local cone

Options:
    -o FILE, --output FILE              Set output file
    -c CONSTR, --constraints CONSTR     Optimization constraints (CHSH|CHSHE)
    -n NUM, --num-runs NUM              Number of searches for each inequality [default: 10]
    -s SUBSET, --select SUBSET          Select subset of inequalities
"""

from operator import matmul
from functools import reduce, partial
import itertools
from math import log2, sin, cos, pi
import cmath
import sys

import scipy.optimize
import numpy as np

from .core.app import application
from .core.symmetry import SymmetryGroup, group_by_symmetry
from .core.io import System, _varset, yaml_dump, format_human_readable
from .core.linalg import (projector, measurement, random_direction_vector,
                          cartesian_to_spherical, kron, to_unit_vector,
                          to_quantum_state, ptrace, ptranspose)


def exp_i(phi):
    return cmath.rect(1, phi)


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


class TripartiteBellScenario(CompositeQuantumSystem):

    symm = 'Aa <> aA; AaBb <> BbAa; AaCc <> CcAa'

    def __init__(self, system):
        super().__init__((2, 2, 2))
        self.cols = system.columns
        sg = SymmetryGroup.load(self.symm, self.cols)
        self.rows = [g[0] for g in group_by_symmetry(sg, system.matrix)]

    def random(self):
        t = np.random.uniform(0, 2*pi)
        d = [
            random_direction_vector(3),
            random_direction_vector(3),
            random_direction_vector(3),
            random_direction_vector(3),
        ]
        s = random_direction_vector(8*2)
        return np.array([t, *np.array(d).flat, *s.flat])

    def unpack(self, params):
        t = params[0]
        d = [[params[1:4], params[4:7]],
             [params[7:10], params[10:13]]]
        s = params[13:]
        angles = [[(0, 0), (t, 0)]]
        angles += [[cartesian_to_spherical(setting)[1:]
                    for setting in party]
                   for party in d]
        state = to_quantum_state(to_unit_vector(s).reshape(8, 2))
        return state, angles

    def realize(self, params):
        state, angles = self.unpack(params)
        parties = [
            [Qbit.rotspin_angles(*setting) for setting in party]
            for party in angles
        ]
        return state, parties

    def _mkcombo(self, parties, cols):
        # measurement combinations
        # initial:  [[(y, n) per angle] per party]        in small hilbert spaces
        # lift:     [[(y, n) per angle] per party]        in large hilbert space
        # select:   [[(ay, an), (by, bn)] per operator pair among different parties]
        # product:  [[(ay,by), (ay,bn), (an,by), (an,bn)] per operator pair]
        # matmul:   [[ay@by, ay@bn, an@by, an@bn] per operator pairing (a,b)]
        _ = self.lift_all(parties)
        _ = select_combinations(_, cols)
        _ = [itertools.product(*m) for m in _]
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


class SEP2(Constraints):

    """The 2-party subsystems are separable."""

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
    system = TripartiteBellScenario(app.system)

    ct = opts['--constraints']
    if ct is None:
        constr = None
    elif ct.upper() == 'CHSHE':
        constr = CHSHE2.optimization_constraints(system)
    elif ct.upper() == 'CHSH':
        constr = CHSH2.optimization_constraints(system)
    elif ct.upper() == 'SEP':
        constr = SEP2.optimization_constraints(system)
    else:
        raise ValueError('Unknown constraints type: {}'.format(ct))

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

            state, angles = system.unpack(result.x)
            yaml_dump([{
                'i': i,
                'coef': expr,
                'cols': system.cols,
                'expr': format_human_readable(expr, system.cols),
                'f': result.fun,
                'state': state,
                'angles': angles
            }], out_file)

            print('\n', i, result.fun)

    print()
