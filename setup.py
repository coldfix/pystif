"""
Install script for pystif.

Usage:
    python setup.py install

Before running this script, install 'cython' and 'numpy'.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    name='pystif',
    version='0.0.0',
    description='Shannon type inequality finder',
    long_description=None,
    author='Thomas Gläßle',
    author_email='t_glaessle@gmx.de',
    url=None,
    license='GPLv3',
    packages=[
        'pystif',
        'pystif.core',
    ],
    ext_modules=cythonize([
        Extension(
            'pystif.core.lp', ['pystif/core/lp.pyx'],
            include_dirs=[numpy.get_include()],
            libraries=['glpk'],
        ),
        Extension(
            'pystif.core.*', ['pystif/core/*.pyx'],
            include_dirs=[numpy.get_include()],
        ),
    ]),
    install_requires=[
        'numpy',
        'scipy',
        'docopt',
    ],
    setup_requires=[
        # In fact, these need to be installed before running setup.py - they
        # are listed here only for documentational purposes:
        'cython',
        'numpy',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
    ],
)
