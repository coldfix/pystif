pystif
======

Collection of utilities to compute projections of high-dimensional convex
cones into lower dimensional subspaces. It works best with integer-only
systems. My primary use-case is the projection of Shannon type cones, which is
why the program may or may not work in other cases.

And by the way it contains a GLPK wrapper API that integrates well with numpy
if you are interested in something like that by itself.

|Tests|


Installation
~~~~~~~~~~~~

First, install these dependencies:

- python≥3.5
- GLPK (development files)
- setuptools (should be installed by default on many systems)
- cython
- numpy
- scipy
- docopt
- funcparserlib

On archlinux, these dependencies can be installed as follows::

    pacman -S python python-setuptools
    pacman -S glpk
    pacman -S cython python-numpy python-scipy
    pip install docopt funcparserlib

To build and install pystif itself, type::

    python setup.py install


Usage
~~~~~

The following subprograms are currently available:

Programs for computing the projection of a convex polyhedron to a subspace:

- ``chm`` — Convex Hull Method
- ``fme`` — Fourier-Motzkin-Elimination
- ``afi`` — Adjacent Facet Iteration
- ``rfd`` — Randomized facet discovery

Peripheral utilities:

- ``equiv`` — check two systems of inequalities for equivalence
- ``makesys`` — create/modify matrix file
- ``pretty`` — human readable display of inequality file
- ``minimize`` — remove redundant inequalities from system

These subprograms are available for execution by their name, e.g.:

.. code-block::

    chm --help

retrieves individual usage information for the ``chm`` utility.

The following typical example computes and prints the marginal entropic
inequalities in a bipartite bell scenario:

.. code-block::

    makesys "rvar A B C D" -o full.txt
    chm full.txt -o small.txt -l1 -s "AC BC AD BD A B C D"
    pretty small.txt -y "AB <> BA; ABCD <> CDAB"

For more examples, see the ``example/`` subdirectory.


Other components
~~~~~~~~~~~~~~~~

There are a few more components which I have currently implemented in C++.
I will try to add the most important functionality here — without
sacraficing too much performance if at all possible.


Conventions
~~~~~~~~~~~

This software operates on convex cones in half-space representation, i.e. the
cone is given by a number of linear constraints of the following standard
form::

    P = {x : Qx ≥ 0} ⊂ ℝⁿ

The constraint matrix ``Q`` can be specified in either of two input formats:

- The first format is a complete listing of all matrix coefficients,
  compatible with ``numpy.loadtxt`` and ``numpy.savetxt``. Each row
  corresponds to one inequality ``q∙x ≥ 0``. Column names are defined in a
  line of the form  ``#:: c1 c2 c3 …``

- The second format is easier to read and write for most humans (presumably)
  as you can specify constraints in the form ``2a + 10b >= c``. It also allows
  to use Shannon information measures (e.g. ``2 I(X:Y|Z) <= H(X)`` and define
  Markov conveniently as ``A -> B -> C -> D``.

For examples look for ``*.txt`` files in the ``example/`` and
``example/data/`` subfolders.

Note that inhomogenious systems can be emulated by thinking of one of the
columns as constant ``1`` (don't you dare thinking of another number!).

When working with entropy spaces the component ``i`` corresponds to the
joint entropy ``H(I)`` of a subset ``I ⊂ {0, …, N-1}`` where the i-th
variable is in ``I`` iff the i-th bit is non-zero in ``i``, e.g.::

    i   bits    I

    0   000     {        }
    1   001     {X₀      }
    2   010     {   X₁   }
    3   011     {X₀,X₁   }
    4   100     {      X₂}
    5   101     {X₀,   X₂}

The zero-th vector component corresponds to the entropy of the empty set which
is defined to be zero. Therefore, that column is usually removed – hence
shifting the meaning of all remaining indices by one. The new 0 column then
corresponds to ``H(X₀)`` etc. To avoid confusion, the column names should
always be annotated using the ``#:: name1 name2 name3 ...`` syntax. The human
readable format avoids this confusion entirely.


.. |Tests| image:: https://api.travis-ci.org/coldfix/pystif.svg?branch=master
   :target: https://travis-ci.org/coldfix/pystif
   :alt: Test status
