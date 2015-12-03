"""
A simple vector class for representing sparse vectors on a dynamic domain.
"""

from collections import abc

from .io import _name


__all__ = [
    'Vector'
]


def _items(of):
    if isinstance(of, abc.Mapping):
        return of.items()
    return of


class Vector(abc.MutableMapping):

    """
    Vector with dynamic set of columns.
    """

    def __init__(self, items=()):
        if isinstance(items, Vector):
            self._data = items._data.copy()
        else:
            self._data = {}
            self += items

    # arithmetic operators
    def __iadd__(self, other):
        for n, c in _items(other):
            self[n] += c
        return self

    def __isub__(self, other):
        for n, c in _items(other):
            self[n] -= c
        return self

    def __imul__(self, scale):
        d = self._data.copy()
        for n in self:
            self[n] *= scale
        return self

    def __add__(self, other):   return Vector(self).__iadd__(other)
    def __sub__(self, other):   return Vector(self).__isub__(other)
    def __mul__(self, scale):   return Vector(self).__imul__(scale)
    def __neg__(self):          return Vector() - self

    # Mapping operations
    def copy(self):             return Vector(self)
    def __iter__(self):         return iter(self._data)
    def __len__(self):          return len(self._data)
    def __bool__(self):         return any(self.values())
    def keys(self):             return self._data.keys()
    def values(self):           return self._data.values()
    def items(self):            return self._data.items()

    def __getitem__(self, key):
        return self._data.get(_name(key), 0)

    # MutableMapping
    def __setitem__(self, key, val):
        name = _name(key)
        if name != 'H()':
            self._data[name] = val

    def __delitem__(self, key):
        del self._data[_name(key)]

    # general operations
    def __str__(self):
        return " + ".join("{} {}".format(v, k)
                          for k, v in self.items()
                          if v != 0)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self)
