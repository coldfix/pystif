"""
Misc utilities.
"""

import numpy as np
from .array import scale_to_int


def call(fn, *args, **kw):
    return fn and fn(*args, **kw)


class OrderedSet:

    """
    Ordered set (=unique list).

    Stuff in the collection must be hashable.
    """

    def __init__(self, iterable=()):
        self._items = list(iterable)
        self._index = {v: i for i, v in enumerate(self._items)}

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __contains__(self, item):
        return item in self._index

    def index(self, item):
        """Find item in the list."""
        return self._index[item]

    def __getitem__(self, index):
        return self._items[index]

    def add(self, item):
        """Add item to the list. Return ``True`` if newly added."""
        if item in self._index:
            return False
        self._items.append(item)
        self._index[item] = len(self._items)-1
        return True


class PointSet(OrderedSet):

    """
    Set of points. You should scale them to integer values on your own before
    putting them in.
    """

    def __init__(self, points=()):
        super(PointSet, self).__init__(tuple(p) for p in points)

    def __contains__(self, point):
        return super(PointSet, self).__contains__(tuple(point))

    def index(self, point):
        return super(PointSet, self).index(tuple(point))

    def add(self, point):
        return super(PointSet, self).add(tuple(point))


class VectorMemory:

    """
    Remember vectors and return if they have been seen.

    Currently works only for int vectors.
    """

    def __init__(self):
        self.seen = set()

    def __call__(self, v):
        v = tuple(scale_to_int(v))
        if v in self.seen:
            return True
        self.seen.add(v)
        return False

    def add(self, *rows):
        for v in rows:
            self(v)


class ExtremePoints(PointSet):

    """
    Utility for finding the extreme points (vertices) of a convex polyhedron
    defined by the projection of a system of linear inequalities.
    """

    def __init__(self, lp, subdim):
        super(ExtremePoints, self).__init__()
        self.lp = lp
        self.subdim = subdim

    @classmethod
    def from_system(cls, system, subdim, limit):
        """Create search problem from the system ``L∙x≥0, x≤limit``."""
        lp = system.lp()
        for i in range(1, subdim):
            lp.set_col_bnds(i, 0, limit)
        return cls(lp, subdim)

    @classmethod
    def inner_approximation(cls, lp, subdim):
        # TODO
        pass

    def search(self, q):
        """
        Search an extreme point ``x`` of the LP which minimizes ``q∙x``.

        The ``q`` parameter must be specified as a vector with ``subdim-1``
        components. Its geometric interpretation is the normal vector of a
        hyperplane in the projection space. This hyperplane is shifted along
        its normal until all points inside the polyhedron fulfill ``q∙x≥0``.

        Returns an extreme point with ``subdim-1`` components.
        """
        assert len(q) == self.subdim-1
        q = np.hstack((0, q))
        # For the random vectors it doesn't matter whether we use `minimize`
        # or `maximize` — but it *does* matter for the oriented direction
        # vectors obtained from other functions:
        extreme_point = self.lp.minimize(q, embed=True)
        extreme_point = extreme_point[1:subdim]
        extreme_point = scale_to_int(extreme_point)
        self.add(extreme_point)
        return extreme_point
