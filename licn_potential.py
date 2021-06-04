#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2021 Johannes Reiff
# SPDX-License-Identifier: MIT

"""
Implementation of the LiCN ⇌ LiNC isomerization potential surface.
This code is based on:

  * R. Essers, J. Tennyson, and P. E. S. Wormer,
    “An SCF potential energy surface for lithium cyanide,”
    Chem. Phys. Lett. 89, 223–227 (1982),
    doi:10.1016/0009-2614(82)80046-8.

  * J. Tennyson,
    “LiCN/LiNC potential surface procedure,”
    Private communication.

Significant parameter deviations between these sources are marked with “!!!”.
"""

import operator
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.special


# Multipole moments <Q_L,0> for the electrostatic energy.
MOMENT_Q = [-1.00, -0.215135, -3.414573, -3.818815, -15.84152, -14.29374, -43.81719]

# Induction energy coefficients C_l1,l2,L.
# Format: (l1, l2): {L: C_l1,l2,L}
# Some parameters include a factor 2 to account for C_l1,l2,L = C_l2,l1,L.
INDUCION_COEFFS = {
    (1, 1): {0: -10.5271, 2: -3.17},
    (2, 1): {1: -20.62328, 3: 3.7320800}, # !!! 4.06 vs 3.7320800
    (2, 2): {0: -57.49396, 2: -106.8192, 4: 17.14139}, # includes (3, 1)
    (3, 2): {1: -202.8972, 3: -75.23207, 5: -28.45514},
    (3, 3): {0: -458.2015, 2: -353.7347, 4: -112.6427, 6: -108.2786},
}

# Damping fit parameters.
DAMPING_R0 = 1.900781
DAMPING_A = 1.515625

# Short-range fit parameters A_L, B_L, and C_L.
SHORT_RNG_PARAMS = [
    # A_L         B_L          C_L
    (-1.3832116, +0.14000706, +0.2078921600),
    (-2.9579132, +1.47977160, -0.0116130820),
    (-4.7420297, +1.81198620, -0.0171806800), # !!! 0.017818 vs 0.0171806800
    (-1.8885299, +1.28750300, +0.0277284910),
    (-4.4143354, +2.32297140, -0.0706927420),
    (-4.0256496, +2.77538320, -0.1377197800),
    (-5.8425899, +3.48085290, -0.1863111400),
    (-2.6168114, +2.65559520, -0.0058815504),
    (-6.3446579, +4.34498010, -0.1529136800),
    (15.2022800, -6.54925370, +1.3025678000),
]


def main():
    minmax_saddle = find_saddle_minmax()
    gradroot_saddle = find_saddle_gradroot()
    min_0, min_pi = find_minima()

    print('MEP maximum:', minmax_saddle, '->', potential(*minmax_saddle))
    print('grad V = 0:', gradroot_saddle, '->', potential(*gradroot_saddle))
    print('min V(ϑ = 0):', min_0, '->', potential(*min_0))
    print('min V(ϑ = π):', min_pi, '->', potential(*min_pi))

    plot_potential(minmax_saddle, min_0, min_pi)


def find_saddle_minmax():
    """Calculate saddle position via a maximum along the minimum energy path."""

    tol = 1e-8
    mep = lambda theta: minimize(potential, 4.5, [float(theta)], tol=tol)
    res = minimize(lambda theta: -mep(theta).fun, 0.8, tol=tol)
    return mep(res.x[0]).x[0], res.x[0]


def find_saddle_gradroot():
    """Calculate saddle position via a root of the potential's gradient."""

    eps = np.sqrt(np.finfo(float).eps)
    pot = lambda pos: potential(*pos)
    dpot = lambda pos: scipy.optimize.approx_fprime(pos, pot, eps)
    res = scipy.optimize.root(dpot, (4.5, 0.8))
    assert res.success
    return tuple(res.x)


def find_minima():
    """Calculate minima corresponding to the LiCN and LiNC configurations."""

    tol = 1e-8
    pot = lambda pos: potential(*pos)
    for theta in (0, np.pi):
        yield tuple(minimize(pot, (4.5, theta), tol=tol).x)


def plot_potential(*points):
    """Show a 2D plot of the potential with markers."""

    r = np.linspace(3.0, 5.5, 256)
    theta = np.linspace(0.0, np.pi, 256)
    mesh_r, mesh_theta = np.meshgrid(r, theta)
    pot = potential(mesh_r, mesh_theta)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.imshow(pot, extent=(r[0], r[-1], theta[0], theta[-1]), vmax=-0.12,
        origin='lower', aspect='auto', interpolation='spline16')
    levels = np.linspace(pot.min(), -0.20, 12)
    ax.contour(mesh_r, mesh_theta, pot, levels, colors='w')
    for point in points:
        ax.plot(*point, marker='x', ms=10, color='tab:orange')

    plt.show()
    plt.close(fig)


def potential(r, theta):
    """Full potential V(R, ϑ)."""

    legendre = [legendre_cos(l, theta) for l in range(len(SHORT_RNG_PARAMS))]
    return (
        (pot_electrostat(r, legendre) + pot_induction(r, legendre)) * damping(r)
        + pot_short_rng(r, legendre)
    )


def pot_electrostat(r, legendre):
    """Electrostatic energy E_el(R, ϑ)."""

    return sum(
        r**(-l - 1) * q * legendre[l]
        for l, q in enumerate(MOMENT_Q)
    )


def pot_induction(r, legendre):
    """Induction energy E_ind(R, ϑ)."""

    return sum(
        r**(-2 - l1 - l2) * sum(c * legendre[l] for l, c in cl.items())
        for (l1, l2), cl in INDUCION_COEFFS.items()
    )


def pot_short_rng(r, legendre):
    """Short-range energy E_SR(R, ϑ)."""

    return sum(map(operator.mul, short_rng_params(r), legendre))


def short_rng_params(r):
    """Short-range parameter D_L(R)."""

    return (np.exp(-a - b * r - c * r**2) for a, b, c in SHORT_RNG_PARAMS)


def damping(r):
    """Damping function F(R)."""

    return 1 - np.exp(-DAMPING_A * (r - DAMPING_R0)**2)


def legendre_cos(l, theta):
    return scipy.special.eval_legendre(l, np.cos(theta))


def minimize(*args, **kwargs):
    res = scipy.optimize.minimize(*args, **kwargs)
    assert res.success
    return res


if __name__ == '__main__':
    sys.exit(int(main() or 0))
