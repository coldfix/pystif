"""
Find quantum violations for the tripartite bell scenario.

Usage:
    qviol EXPRS [-c CONSTR] [-o FILE] [-n NUM]

Options:
    -o FILE, --output FILE              Set output file
    -c CONSTR, --constraints CONSTR     Optimization constraints (CHSH|CHSHE)
    -n NUM, --num-runs NUM              Number of searches for each inequality [default: 10]
"""

from operator import matmul, mul
from functools import reduce, partial
import itertools
from math import log2, sin, cos, pi, sqrt, acos, atan2
import cmath
import sys

import scipy.optimize
import numpy as np
from docopt import docopt
import yaml

from .core.symmetry import SymmetryGroup, group_by_symmetry
from .core.io import System, _varset


def dagger(M):
    """
    Return transpose conjugate.
    """
    return M.conj().T


def kron(*parts):
    """Compute the repeated Kronecker product (tensor product)."""
    return reduce(np.kron, parts)


def cartesian_to_spherical(v):
    """
    Coordinate transformation from Cartesian to spherical.

    For an input vector of the form

        r sin(θ) cos(φ)
        r sin(θ) sin(φ)
        r cos(θ)

    Returns (r, theta, phi).
    """
    r = np.linalg.norm(v)
    theta = acos(v[2]/r)
    phi = atan2(v[1], v[0])
    return (r, theta, phi)


def to_unit_vector(v):
    """Normalize a cartesian vector."""
    return v / np.linalg.norm(v)


def random_direction_angles():
    """Return unit vector on the sphere in spherical coordinates (θ, φ)."""
    v = np.random.normal(size=3)
    r, theta, phi = cartesian_to_spherical(v)
    return theta, phi


def random_direction_vector(size):
    """Return unit vector on the sphere in cartesian coordinates (x, y, z)."""
    return to_unit_vector(np.random.normal(size=size))


def complex2real(z: complex, eps=1e-13) -> float:
    """
    Convert a complex number to a real number.

    Use this function after calculating the expectation value of a hermitian
    operator.
    """
    if z.imag > eps:
        raise ValueError("{} is not a real number.".format(z))
    return z.real


def expectation_value(psi, M) -> complex:
    """
    Return the expectation value <ψ|M|ψ>.

    :param np.ndarray psi: vector m*1
    :param np.ndarray M: matrix m*m
    """
    return dagger(psi) @ M @ psi


def measurement(psi, M) -> float:
    """Return the measurement <ψ|M|ψ> of a hermitian operator."""
    return complex2real(expectation_value(psi, M))


def as_column_vector(vec):
    """Reshape the array to a column vector."""
    vec = np.asarray(vec)
    return vec.reshape((vec.size, 1))


def projector(vec):
    """
    Return the projection matrix to the 1D space spanned by the given vector.
    """
    vec = as_column_vector(vec)
    return vec @ dagger(vec)


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
        exp_th2 = cmath.rect(1, theta/2)
        exp_phi = cmath.rect(1, phi)
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
        state = to_unit_vector(s)
        state = np.array([complex(a, b) for a, b in zip(state[::2], state[1::2])])
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


def yaml_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class _Dumper(Dumper):
        pass
    def numpy_scalar_representer(dumper, data):
        return dumper.represent_data(np.asscalar(data))
    def numpy_array_representer(dumper, data):
        return dumper.represent_data([x for x in data])
    def complex_representer(dumper, data):
        return dumper.represent_data([data.real, data.imag])
    _Dumper.add_multi_representer(np.generic, numpy_scalar_representer)
    _Dumper.add_representer(np.ndarray, numpy_array_representer)
    _Dumper.add_representer(complex, complex_representer)
    return yaml.dump(data, stream, _Dumper, **kwds)


def main(args=None):

    opts = docopt(__doc__, args)

    system = TripartiteBellScenario(System.load(opts['EXPRS']))

    ct = opts['--constraints']
    if ct is None:
        constr = None
    elif ct.upper() == 'CHSHE':
        constr = CHSHE2.optimization_constraints(system)
    elif ct.upper() == 'CHSH':
        constr = CHSH2.optimization_constraints(system)
    else:
        raise ValueError('Unknown constraints type: {}'.format(ct))

    num_runs = int(opts['--num-runs'])

    if opts['--output']:
        out_file = open(opts['--output'], 'wt')
    else:
        out_file = sys.stdout

    for _ in range(num_runs):
        for i, expr in enumerate(system.rows):
            result = scipy.optimize.minimize(
                system.violation, system.random(),
                (expr,), constraints=constr)

            if result.fun > -1e-11:
                print('.' if result.success else 'x', end='', flush=True)
                continue

            state, angles = system.unpack(result.x)
            yaml_dump([{
                'i': i,
                'f': result.fun,
                'state': state,
                'angles': angles
            }], out_file)

            print("\n", i, expr)
            print('\n', result.fun, sep='')
            print('\n', result.x, sep='')

    print()


if __name__ == '__main__':
    main()
