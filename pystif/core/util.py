"""
Misc utilities.
"""

from functools import wraps

from .array import scale_to_int


def call(fn, *args, **kw):
    return fn and fn(*args, **kw)


def cached(fn):
    key = '_' + fn.__name__
    @wraps(fn)
    def wrapper(self):
        try:
            return getattr(self, key)
        except AttributeError:
            val = fn(self)
            setattr(self, key, val)
            return val
    return wrapper


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
