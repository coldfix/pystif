"""
Plot results for the AFI/CHM benchmarks.
"""

from operator import itemgetter as _C
from itertools import groupby
import sys
import math

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import numpy.lib.recfunctions

from fit import Fit, format_uncert, format_scientific


def load_data(filename):
    data = np.loadtxt(filename, dtype=[
        #'dim', 'num_rows',
        ('subdim', int),
        ('t_afi', float), #'ret_afi',
        ('t_chm', float), #'ret_chm',
        ('num_facets', int),
        ('num_ridges', int),
        ('num_vertices', int),
        ('ridges_per_facet', float),
        ('vertices_per_facet', float),
        ('vertices_per_ridge', float),
        ('vpf_geo', float),
    ])
    data = np.rec.array(data)
    data = np.lib.recfunctions.rec_append_fields(
        data, ['afi_cmplx', 'chm_cmplx'], [
            data.vertices_per_facet ** ((data.subdim-1)/2) * data.num_facets,
            data.num_vertices ** (data.subdim/2),
        ]
    )
    return data


def rec(x):
    """Convert iterable of structured arrays to structured array."""
    return np.rec.array(np.array(list(x)))


def rec_groupby(data, key):
    """Group the rows of a structured array using the given key function. The
    result is a list of structured arrays."""
    return [rec(g) for k, g in groupby(sorted(data, key=key), key)]


def rec_mean(x):
    """Compute the mean of the columns of a structured array. The result is a
    structured array of the mean values."""
    return np.rec.array(
        np.mean([x[n] for n in x.dtype.names], axis=1).T,
        dtype=[(n, float) for n in x.dtype.names])


def rec_group_mean(x, key):
    """Group the values of a structured array by the given column and compute
    the mean value within each group. The result is returned as a structured
    array."""
    return rec(map(rec_mean, rec_groupby(x, key))).flatten()


class AFI_Model:

    init = [5e-5, 0.5, 0.5]

    def __call__(self, x, a, b, c):
        return (
            a * x.num_facets * x.num_vertices**((x.subdim-1)*b)
            + c * x.num_facets * x.ridges_per_facet * np.log2(x.num_vertices)
        )

    def model(self, a, b, c):
        return r'${}\ f\ v^{{(d-1)/{:.3}}}{}+ {}\ f\ \overline{{r_f}}\ \log_2 v$'.format(
            format_scientific(a),
            1/b,
            "$\n$\\ ",
            format_scientific(c))


class AFI_Model2:

    #init = [5e-5, 0.5, 0.5, 1]
    init = [5e-5, 0.5]

    def __call__(self, x, a, b, c=0.5, d=1):
        return (
            a * x.num_facets * x.vpf_geo**((x.subdim-d)*c)
            + b * x.num_facets * x.ridges_per_facet * np.log2(x.num_vertices)
        )

    def model(self, a, b, c=0.5, d=1):
        return r'${0}\ f\ v^{{(d-{3})/{2:.3}}} + {1}\ f\ \overline{{r_f}}\ \log_2 v$'.format(
            format_scientific(a),
            format_scientific(b),
            1/c, d
        )



class CHM_Model:

    init = [5e-6, 0.6, +1, 1e-3, 0]

    def __call__(self, x, a, b, c, d=0, e=0, f=1):
        return (a * x.num_vertices**(b*x.subdim)
                * np.log(x.num_vertices) / np.log(x.subdim + f)
                + d * x.num_vertices
                #+ f
                )

    def model(self, a, b, c, *stuff):
        return r'CHM TODO'
        return r'${}\ f\ v^{{(d-1)/{:.3}}} + {}\ r\ \log_2 v$'.format(
            format_scientific(a),
            1/b,
            format_scientific(c))



class Plotter:

    STR_STYLE = dict(ms=4, ls="none", marker="o", color="#ffff00")
    AFI_STYLE = dict(ms=4, ls="none", marker="v", color="#ff0000", label="AFI")
    CHM_STYLE = dict(ms=4, ls="none", marker="o", color="#ffff00", label="CHM")

    LABEL = {
        'chm_cmplx_': "estimated CHM complexity $v^{d/2}$",
        'afi_cmplx_': "estimated AFI complexity $f \cdot v_f^{(d-1)/2}$",
        'chm_cmplx': "structure constant $v^{d/2}$",
        'afi_cmplx': "structure constant $f \cdot v_f^{(d-1)/2}$",
        'runtime': "method runtime [s]",
        'runtime_actual': "actual runtime [s]",
        'runtime_fitted': "fitted runtime [s]",
    }

    def __init__(self, data):
        matplotlib.rcParams.update({'text.usetex': True})
        self._data = data
        self.select()

    def _figure(self, xaxis=None, yaxis=None, figsize=(5.5, 3.5)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if xaxis is not None:
            ax.set_xlabel(self.LABEL[xaxis])
        if yaxis is not None:
            ax.set_ylabel(self.LABEL[yaxis])
        fig.tight_layout()
        return fig, ax

    def select(self, mindim=None, maxdim=None, groupby=None):
        data = self._data
        if mindim is not None: data = data[data.subdim >= mindim]
        if maxdim is not None: data = data[data.subdim <= maxdim]
        if groupby: rec_groupby(data, groupby)
        self.data = data
        self.fitted = False

    def geometry(self):
        x = self.data
        fig, ax = self._figure('chm_cmplx_', 'afi_cmplx_')
        ax.loglog(x.chm_cmplx, x.afi_cmplx, **self.STR_STYLE)
        return fig

    def runtimes(self, xaxis='chm_cmplx'):
        x = self.data
        #fig, ax = self._figure(xaxis, 'runtime')
        fig, ax = self._figure()
        ax.loglog(x[xaxis], x.t_afi, **self.AFI_STYLE)
        ax.loglog(x[xaxis], x.t_chm, **self.CHM_STYLE)
        ax.legend(loc='upper left', numpoints=5)
        return fig

    def _fit(self, afi_model=AFI_Model(), chm_model=CHM_Model(), force=False):
        if self.fitted and not force:
            return
        self.fitted = True

        def yerror(t):
            return np.sqrt(t)

        data = self.data
        self.afi_fit = Fit.leastsq(
            afi_model, data,
            data.t_afi, yerror(data.t_afi),
            afi_model.init)
        print(self.afi_fit.info())

        self.chm_fit = Fit.leastsq(
            chm_model, data,
            data.t_chm, yerror(data.t_chm),
            chm_model.init)
        print(self.chm_fit.info())

        self.model = {
            'afi': afi_model.model(*self.afi_fit.coeffs),
            'chm': chm_model.model(*self.chm_fit.coeffs),
        }


    def runtime_fit(self, xaxis='chm_cmplx'):
        self._fit()

        data = self.data
        yfit = self.afi_fit(data)
        zfit = self.chm_fit(data)
        x = data[xaxis]

        fig, ax = self._figure(xaxis, 'runtime')

        ax.loglog(x, yfit, color="#ffff00", ls="none", marker=">",
                  label="fitted AFI runtime")
        ax.loglog(x, zfit, color="#00ff00", ls="none", marker="<",
                  label="fitted CHM runtime")

        ax.loglog(x, data.t_afi, ms=4, color="r", ls="none", marker="v",
                  label="actual AFI runtime")
        ax.loglog(x, data.t_chm, ms=4, color="b", ls="none", marker="o",
                  label="actual CHM runtime")

        ax.legend(loc='upper left', numpoints=5)
        return fig

    def fit_versus_actual(self, which='afi'):
        self._fit()

        data = self.data
        fit = getattr(self, which+'_fit')(self.data)

        fig, ax = self._figure(figsize=(5.5,3))
        ax.loglog(data.t_afi, fit, color="blue", ls="none", marker="o",
                  label="$"+self.model[which][1:], ms=2)
        ax.loglog(fit, fit, '-', color="red", lw=1.5, label="$y=x$")
        legend = ax.legend(loc='upper left', numpoints=5,
                           #bbox_to_anchor=(0.00, 0.99),
                           bbox_transform=ax.transAxes,
                           #borderaxespad=0
                           )
        legend.get_frame().set_alpha(0)

        num = max(max(fit), max(data.t_afi))
        exp = math.log10(num)
        new = (math.ceil(num * math.pow(10, -exp)) + 1) * math.pow(10, exp)
        ax.set_xlim(right=new)

        return fig

    def vpf_vert(self):
        fig, ax = self._figure('runtime', 'runtime')
        ax.plot(self.data.num_vertices, self.data.vpf_geo, 'o')
        return fig

def main():
    p = Plotter(load_data(sys.argv[1]))

    # IN THESIS:
    p.runtimes('afi_cmplx').savefig("runtime-all.pdf")
    #p._fit(AFI_Model2())
    p.fit_versus_actual('afi').savefig("fit-versus-actual-afi.pdf")

    # OTHERS:
    p.select(mindim=10)
    p.fit_versus_actual('chm').savefig("fit-versus-actual-chm.pdf")

    return

    p.vpf_vert().savefig('bla.pdf')
    p.runtime_fit('afi_cmplx').savefig("fit-afi_cmplx.pdf")
    return

    p.select(mindim=10)
    p.runtimes('afi_cmplx').savefig("time__afi-hi.pdf")
    p.runtimes('chm_cmplx').savefig("time__chm-hi.pdf")

    p.select(maxdim=10)
    p.runtime_fit('afi_cmplx').savefig("fit-afi_cmplx-lowdim.pdf")
    p.fit_versus_actual('afi').savefig("fit-versus-actual-afi-lowdim.pdf")
    p.fit_versus_actual('chm').savefig("fit-versus-actual-chm-lowdim.pdf")

    return

    p.geometry().savefig("polytope-geometry.pdf")


    p.select(mindim=9)
    p.runtime_fit('afi_cmplx').savefig("runtime-chm-afi-YYY.pdf")
    p.runtime_fit('chm_cmplx').savefig("runtime-chm-afi-max.pdf")

    p.select(mindim=8)
    p.runtimes('afi_cmplx').savefig("runtime-chm-afi-xxx.pdf")

    p.select(maxdim=9)
    p.runtimes('afi_cmplx').savefig("time__afi-low.pdf")
    p.runtimes('chm_cmplx').savefig("time__chm-low.pdf")

    p.select(mindim=12)
    p.runtimes('chm_cmplx').savefig("runtime-chm-afi-max.pdf")

    p.select(maxdim=12)
    p.runtimes('chm_cmplx').savefig("runtime-chm-afi-all.pdf")

    plt.show()


if __name__ == '__main__':
    main()
