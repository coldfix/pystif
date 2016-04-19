# encoding: utf-8
"""
Utils for fitting.
"""
__all__ = ['format_uncert',
           'normed_residual',
           'ConvergenceError',
           'Fit']

#----------------------------------------
# Useful fit functions to keep in mind:
#
# linear:       scipy.stats.linregress
# polynomial:   numpy.polyfit
# general:      scipy.optimize.curve_fit
#               scipy.optimize.leastsq
#               numpy.linalg.lstsq
#----------------------------------------

import math
from functools import partial, wraps

import numpy as np
from scipy.optimize import leastsq, minimize


def cachedproperty(func):
    """A memoize decorator for class properties."""
    key = '_' + func.__name__
    @wraps(func)
    def get(self):
        try:
            return getattr(self, key)
        except AttributeError:
            val = func(self)
            setattr(self, key, val)
            return val
    return property(get)


def _sign(number):
    return "{-}" if number < 0 else ""


def format_scientific(value):
    dlog = math.floor(math.log10(value))
    power = math.pow(10, -dlog)
    value *= power
    if dlog == 0:
        return r'{%s}{%.1f}' % (_sign(value), abs(value))
    else:
        return r'{%s}{%.1f}{\scriptstyle\mathrm{E}}^{%s%d}' % (
            _sign(value), abs(value),
            _sign(dlog), abs(dlog))


def format_uncert(value, uncert):
    """
    LaTeX scientific notation of a value with uncertainty.

    :param float value: value
    :param float uncert: uncertainty
    :returns: LaTeX formatted

    """
    dlog = math.floor(math.log10(uncert))
    power = math.pow(10, -dlog)
    value *= power
    uncert *= power
    if dlog == 0:
        return r'(%.1f \pm %.1f)' % (value, uncert)
    else:
        return r'(%.1f \pm %.1f) \times 10^{%d}' % (value, uncert, dlog)

class ConvergenceError(RuntimeError):
    """Raised when a fit does not converge."""

def normed_residual(f, x, y, y_err, coeffs):
    """Calculate the normed residual `(f(x, *coeffs) - y) / y_err`."""
    return (y - f(x, *coeffs)) / y_err

class Fit(object):
    """
    Provides fit results for the equation `f(x, *coeffs) = y`.
    """
    def __init__(self, f, x, y, y_err, coeffs, covariance_matrix):
        self._f = f
        self._x = x
        self._y = y
        self._y_err = y_err
        self._coeffs = coeffs
        self._covariance_matrix = covariance_matrix

    @cachedproperty
    def dof(self):
        """Number of degrees of freedom."""
        return len(self._x) - len(self._coeffs)

    @property
    def coeffs(self):
        """Fitted coefficients."""
        return self._coeffs

    @cachedproperty
    def residuals(self):
        """Normed residuals."""
        return normed_residual(self._f, self._x, self._y, self._y_err,
                               self._coeffs)

    @cachedproperty
    def chisq(self):
        """Square sum of residuals χ²."""
        r = self.residuals
        return sum(r * r)

    @cachedproperty
    def reduced_chisq(self):
        """Reduced χ²."""
        return self.chisq / self.dof

    @cachedproperty
    def stderr(self):
        """
        Standard errors of fit coefficients.

        These are calculated as per gnuplot, "fixing" the result for non
        unit values of the reduced chisq.

        """
        if self._covariance_matrix is None:
            return -1
        cov_diag = self._covariance_matrix.diagonal()
        return np.sqrt(cov_diag * self.reduced_chisq)

    def info(self):
        """Return info about fit results."""
        return ("chisq = {0.chisq}\n"
                "dof = {0.dof}\n"
                "reduced chisq = {0.reduced_chisq}\n"
                "coeffs = {0.coeffs}\n"
                "stderr = {0.stderr}\n").format(self)

    @classmethod
    def leastsq(cls, f, x, y, y_err, coeffs, xmin=None, xmax=None, bounds=None):
        """
        Perform Levenberg-Marquardt fit for `f(x) = y`.

        :param callable f: model curve
        :param array x: x values of measured data
        :param array y: y values of measured data
        :param array y_err: uncertainty in y values
        :param array coeffs: initial guess for coefficients
        :param float xmin: restrict x range
        :param float xmax: restrict x range
        :returns: fit results
        :rtype: Fit
        :raises ConvergenceError: if the fit does not converge
        """
        if xmin is not None or xmax is not None:
            mask = True
            if xmin is not None:
                mask &= x >= xmin
            if xmax is not None:
                mask &= x <= xmax
            x = x[mask]
            y = y[mask]
            y_err = y_err[mask]

        if bounds is not None:
            bounds = np.array(bounds)
            bf1 = lambda params: params - bounds[:,0]
            bf2 = lambda params: bounds[:,1] - params
            def optfunc(params):
                return np.sum(normed_residual(f, x, y, y_err, params)**2)
            result = minimize(
                optfunc,
                np.array(coeffs),
                bounds=bounds, 
                # constraints=[
                    # {'type': 'ineq', 'fun': bf1},
                    # {'type': 'ineq', 'fun': bf2},
                # ]
            )
            print(result)
            coeffs = result.x
            return cls(f, x, y, y_err, result.x, None)

        result = leastsq(partial(normed_residual, f, x, y, y_err),
                         coeffs, full_output=True)
        coefficients, covariance_matrix, info, message, success = result
        if success not in (1, 2, 3, 4):
            raise ConvergenceError(message)
        return cls(f, x, y, y_err, coefficients, covariance_matrix)

    def __call__(self, x):
        return self._f(x, *self.coeffs)
