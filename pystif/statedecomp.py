"""
Classify 3-qubit states.

Usage:
    statedecomp FILENAME


This is an implementation of the algorithm presented in:

    A. Acı́n, A. Andrianov, E. Jané and R. Tarrach (2000)
    "Three-qubit pure-state canonical forms".
    http://arxiv.org/abs/quant-ph/0009107
"""

import cmath
import math

import numpy as np

import yaml

from docopt import docopt

from .core.linalg import (projector, dagger, kron, complex2real,
                          solve_quadratic_equation, hdet, ptrace,
                          random_quantum_state)


def to_canonical_form(s):

    """
    Compute the canonical form of a 3-qubit state (given as a 8D flat vector).
    """

    # Demand 0 = det(T₀') = det(T₀ + xT₁) where x = U₀₁/U₀₀:
    x0, x1 = solve_quadratic_equation(
        s[4]*s[7] - s[6]*s[5],
        s[4]*s[3] + s[0]*s[7] - s[2]*s[5] - s[6]*s[1],
        s[0]*s[3] - s[2]*s[1],
    )

    s0 = _tcf(s, x0)
    s1 = _tcf(s, x1)
    phi0 = cmath.phase(s0[4])

    if np.isclose(phi0, 0) or np.isclose(phi0, math.pi):
        if np.isclose(abs(s0[4]), abs(s1[4])):
            return s0 if s0[0] < s1[0] else s1
        return s0 if s0[4] < s1[4] else s1
    return s0 if phi0 > 0 and phi0 < math.pi else s1


def _tcf(s, x):

    # 1 - |a|² = |b|² = |xa|²  =>  |a|² = 1/(|x|² + 1)
    a = math.sqrt(1 / (abs(x)**2 + 1))
    b = a * x
    b_ = b.conjugate()
    u_A = np.array([
        [ a,  b],
        [-b_, a],
    ])

    #t0, t1 = np.tensordot(u_A, [t0, t1], 1)
    u_A = kron(u_A, np.eye(4))

    s1 = u_A @ s
    t = s1.reshape((2, 2, 2))

    # Now diagonalize  T₀ = U^† S V^†   <=>     D = U T₀ V
    #                                   <=>     |ψ'> = (1 ⊗ U ⊗ V^T) |ψ>
    U_, S, V_ = np.linalg.svd(t[0])
    u_BC = kron(np.eye(2), dagger(U_), dagger(V_).T)

    s2 = u_BC @ s1

    # Next, absorb phases into |0>_A, |1>_A, |1>_B, |1>_C:

    phi = {i: cmath.phase(s2[i])
           for i in (0, 5, 6, 7)}

    A0 = cmath.rect(1, -phi[0])
    A1 = cmath.rect(1, phi[7] - phi[5] - phi[6])
    B = cmath.rect(1, phi[5] - phi[7])
    C = cmath.rect(1, phi[6] - phi[7])

    U_phase = np.array([
        [[A0, 0],
         [0, A1]],
        [[1, 0],
         [0, B]],
        [[1, 0],
         [0, C]],
    ])

    s3 = kron(*U_phase) @ s2

    assert np.allclose(0, s3[1:4])
    assert np.allclose(0, [
        cmath.phase(x) for x in s3[[0, 5, 6, 7]]
    ])

    return s3


def polynomial_invariants_I(state):
    """
    Compute polynomial invariants due to

        A. Sudbery (2001), "On local invariants of pure three-qubit states"
        http://arxiv.org/abs/quant-ph/0001116
    """
    dims = (2, 2, 2)
    rho_ABC = projector(state)
    rho_A = ptrace(rho_ABC, dims, 1, 2)
    rho_B = ptrace(rho_ABC, dims, 0, 2)
    rho_C = ptrace(rho_ABC, dims, 0, 1)
    rho_AB = ptrace(rho_ABC, dims, 2)
    I = [
        np.trace(rho_A @ rho_A),
        np.trace(rho_B @ rho_B),
        np.trace(rho_C @ rho_C),
        np.trace(kron(rho_A, rho_B) @ rho_AB),
        abs(hdet(state.reshape(dims)))**2,
    ]
    return [complex2real(i) for i in I]


def polynomial_invariants_J(state):
    I = polynomial_invariants_I(state)
    SI4 = math.sqrt(I[4])
    return [
        (1 + I[0] - I[1] - I[2] - 2*SI4) / 4,
        (1 - I[0] + I[1] - I[2] - 2*SI4) / 4,
        (1 - I[0] - I[1] + I[2] - 2*SI4) / 4,
        SI4,
        (3 - 3*I[0] - 3*I[1] - I[2] + 4*I[3] - 2*SI4) / 4,
    ]


def polynomial_invariants_J_verify(state):
    """Alternative computation of the J invariants."""
    state = to_canonical_form(state)
    lam = [state[i].real for i in (0, 4, 5, 6, 7)]
    mu = [l**2 for l in lam]
    J0 = abs(lam[4]*state[4] - lam[2]*lam[3])**2
    return [
        J0,
        mu[0]*mu[2],
        mu[0]*mu[3],
        mu[0]*mu[4],
        mu[0]*(J0 + mu[2]*mu[3] - mu[1]*mu[4]),
    ]


def classify_state(state):
    J = polynomial_invariants_J(state)
    zero = np.isclose(J, 0)
    numz = sum(zero)

    # pure state
    if numz == 5:
        return 1, 'a'

    # entanglement between 2 qubits
    if numz == 4 and any((zero[0], zero[1], zero[2])):
        return 2, 'a'
    if numz == 4 and zero[3]:
        return 2, 'b'

    a = J[0]*J[1] + J[0]*J[2] + J[1]*J[2]
    b = math.sqrt(J[0]*J[1]*J[2])
    c = J[4]/2
    d = J[0]*J[3] + J[0]*J[1] + J[0]*J[2] + J[1]*J[2]

    # 3 terms
    if numz == 1 and zero[3] and np.isclose(a, b) and np.isclose(b, c):
        return 3, 'a'

    # 4 terms
    if numz == 1 and zero[3]:
        return 4, 'a'
    if numz == 2 and zero[4] and (zero[1] or zero[2]):
        return 4, 'b'
    if numz == 0 and np.isclose(b, c) and np.isclose(c, d):
        return 4, 'c'

    raise AssertionError("Unknown state class")


def test_classify():
    s = random_quantum_state(8)
    to_canonical_form(s)
    print(polynomial_invariants_J(s))
    print(polynomial_invariants_J_verify(s))
    print([a - b for a, b in zip(polynomial_invariants_J(s), polynomial_invariants_J_verify(s))])
    assert np.allclose(
        polynomial_invariants_J(s),
        polynomial_invariants_J_verify(s),
    )


def main(args=None):
    opts = docopt(__doc__, args)

    with open(opts['FILENAME']) as f:
        data = yaml.safe_load(f)

    for entry in data:
        state = np.array([
            complex(a, b)
            for a, b in entry['state']
        ])
        assert np.allclose(
            polynomial_invariants_J(state),
            polynomial_invariants_J_verify(state),
        )

        s = to_canonical_form(state)
        lam = [s[i].real for i in (0, 4, 5, 6, 7)]
        phi = cmath.phase(s[4])
        print(lam, phi)


if __name__ == '__main__':
    main()
