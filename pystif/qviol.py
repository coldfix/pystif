"""
Small script to find quantum violations

Usage:
    qviol EXPRS [-y SYM]

Options:
    -y SYM, --symmetry SYM                  Specify symmetry group generators
"""

from operator import matmul, mul
from functools import reduce
import itertools
from math import log2, sin, cos, pi

import scipy.optimize
import numpy as np
from docopt import docopt

from .core.symmetry import SymmetryGroup
from .core.util import scale_to_int


def dagger(M):
    """
    Return transpose conjugate.
    """
    return M.conj().T


def complex2real(z):
    assert abs(z.imag) < 1e-13
    return z.real


def complement(P):
    """
    Return (id - P).

    :param np.ndarray P: projection operator
    """
    dim = P.shape[0]
    return np.eye(dim) - P


def expectation_value(state, M):
    """
    Return expectation value of the observable

    :param np.ndarray state: vector m*1
    :param np.ndarray M: matrix m*m
    """
    return dagger(state) @ M @ state


def entropy(state, measurements):
    return H([
        complex2real(expectation_value(state, m))
        for m in measurements
    ])


class Qbit:

    dim = 2

    sigma_x = np.array([[0,   1], [ 1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1,   0], [0, -1]])
    sigma = np.array((sigma_x, sigma_y, sigma_z))

    @classmethod
    def rotspin(cls, direction):
        spinmat = np.dot(cls.sigma.T, direction).T
        # eigenvalues/-vectors of hermitian matrix, in ascending order:
        val, vec = np.linalg.eigh(spinmat)
        spin_down_projector = vec[:,[0]] @ dagger(vec[:,[0]])
        spin_up_projector = vec[:,[1]] @ dagger(vec[:,[1]])
        assert np.allclose(spin_down_projector + spin_up_projector, np.eye(2))
        return spin_down_projector

    @classmethod
    def xzspin(cls, angle):
        return cls.rotspin([sin(angle), 0, cos(angle)])


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
        return np.kron(np.kron(l, operator), r)

    def lift_all(self, parties):
        """
        Generate measurements from
        """
        return [[(self.lift(m, i),
                  self.lift(complement(m), i))
                 for m in povms]
                for i, povms in enumerate(parties)]


def group_by_symmetry(sg, vectors):
    groups = []
    belong = {}
    for row in vectors:
        row_ = tuple(scale_to_int(row))
        if row_ in belong:
            belong[row_].append(row)
            continue
        group = [row]
        groups.append(group)
        for sym in sg(row):
            sym_ = tuple(scale_to_int(sym))
            belong[sym_] = group
    return groups


# composed operations
def h(p):
    """Compute one term in the entropy sum."""
    return -p*log2(p) if p > 0 else 0

def H(pdist):
    """Compute entropy from a numpy array of probabilities."""
    assert abs(sum(pdist) - 1) < 1e-10
    return sum(map(h, pdist))


def mkcombo(parties):
    return [combo
            for a, b in itertools.combinations(parties, 2)
            for combo in itertools.product(a, b)]


def violation(state, expr, mcombo):
    state = np.hstack((1, 0, state))
    state = state/np.linalg.norm(state)
    state = np.array([complex(a, b) for a, b in zip(state[::2], state[1::2])])
    entropies = [entropy(state, m) for m in mcombo]
    return np.dot(entropies, expr)


def random_direction_pair():
    d = np.random.normal(size=3)
    d /= np.linalg.norm(d)
    n = np.random.normal(size=2)
    s = d[:2] @ n
    n = np.hstack((n, -s/d[2]))
    n /= np.linalg.norm(n)
    return d, n


def main(args=None):

    opts = docopt(__doc__, args)

    system = CompositeQuantumSystem((2, 2, 2))

    # set up measurements
    alpha = (0, 3*pi/4, 6*pi/4)
    #alpha = (0, 0, 0)
    #parties = [[Qbit.xzspin(a), Qbit.xzspin(a+pi/2)] for a in alpha]

    d1, n1 = random_direction_pair()
    d2, n2 = random_direction_pair()
    parties = [
        [Qbit.xzspin(0), Qbit.xzspin(pi/2)],
        [Qbit.rotspin(d1), Qbit.rotspin(n1)],
        [Qbit.rotspin(d2), Qbit.rotspin(n2)],
    ]

    # measurement combinations
    mcols = mkcombo(['Aa', 'Bb', 'Cc'])
    # initial:  [[a, b] per party]
    # lift:     [[(y, n) per angle] per party]
    # mkcombo:  [[(ay, an), (by, bn)] per operator pairing among different parties]
    # product:  [[(ay,by), (ay,bn), (an,by), (an,bn)] per operator pairing (a,b)]
    # matmul:   [[ay@by, ay@bn, an@by, an@bn] per operator pairing (a,b)]
    mcombo = mkcombo(system.lift_all(parties))
    mcombo = [itertools.product(*m) for m in mcombo]
    mcombo = [[reduce(matmul, parts) for parts in m] for m in mcombo]

    # TODO: rearrange + discard symmetries
    exprs = np.loadtxt(opts['EXPRS'])
    if opts['--symmetry']:
        sg = SymmetryGroup.load(opts['--symmetry'], mcols)
        groups = group_by_symmetry(sg, exprs)
        # take one representative from each category:
        exprs = [g[0] for g in groups]

    c = 0
    expr = exprs[13]
    for i in range(50):
        for expr in exprs:
            state = np.random.normal(size=2*system.dim-2)
            result = scipy.optimize.minimize(violation, state, (expr, mcombo))
            if result.success:
                print(result.fun)
                if result.fun < 0:
                    print(result)
                    c += 1
            else:
                print(result.message)
                print(result.x)

    print(c, "/", len(exprs))


if __name__ == '__main__':
    main()
