"""
Plot mean+variance of the output of thhe benchmark.py script.

Usage:
    plot.py RESULTS_FILE
"""

from matplotlib import pyplot as plt
from itertools import groupby
import numpy as np
import sys

data = np.loadtxt(sys.argv[1])
cols = [
    'dim', 'num_rows', 'subdim',
    't_afi', 'ret_afi',
    't_chm', 'ret_chm',
    'num_facets',
    'num_ridges',
    'num_vertices',
    'ridges_per_facet',
    'vertices_per_facet',
    'vertices_per_ridge',
]

def _i(*args):
    return [cols.index(c) for c in args]

key = lambda row: row[0]
rows = data[:,_i('subdim', 't_afi', 't_chm')]

dat = [list(g) for k, g in groupby(rows, key)]
dat = [(*np.mean(g, axis=0), *np.std(g, axis=0)) for g in dat]
dat = np.array(dat)

ax = plt.subplot(111)
ax.set_xlim(4.5, 12.5)
ax.set_yscale('log')
ax.errorbar(dat[:,0], dat[:,1], yerr=dat[:,4], fmt='og', ms=10, label="AFI")
ax.errorbar(dat[:,0], dat[:,2], yerr=dat[:,5], fmt='vr', ms=10, label="CHM")

ax.legend(loc='upper left')

plt.savefig('runtime.pdf')
plt.show()
