from functools import partial
from os import path
import sys
import re

import numpy as np
import yaml

from .array import format_vector
from .lp import Problem
from .util import VectorMemory


def _varset(key):
    if isinstance(key, (set,list,tuple)):
        return set(key)
    if key.startswith('H(') and key.endswith(')'):
        return set(re.split('[ ,]', key[2:-1]))
    if key.startswith('_'):
        return set(key[1:])
    raise ValueError("Unknown format.")


def varsort(varnames):
    return sorted(varnames, key=lambda s: (s.lower(), s))


def _name(key):
    try:
        return 'H(' + ','.join(varsort(_varset(key))) + ')'
    except ValueError:
        return key


def detect_prefix(s, prefix, on_prefix):
    """
    Check whether ``s`` starts with ``prefix``. If so, call ``on_prefix`` with
    the stripped string.
    """
    if s.startswith(prefix):
        if on_prefix:
            on_prefix(s[len(prefix):])
        return True
    return False


def remove_comments(lines, on_comment=None):
    """Iterate over non-comment lines. Forward comments to ``on_comment``."""
    for line in lines:
        if not detect_prefix(line.strip(), '#', on_comment):
            yield line


def read_system_from_file(file):
    lines = list(file)
    try:
        return read_table_from_file(lines)
    except ValueError:
        pass
    from .parse import parse_text
    return parse_text("\n".join(lines))


def read_table_from_file(file):
    comments = []
    contents = remove_comments(file, comments.append)
    matrix = np.loadtxt(contents, ndmin=2)
    cols = []
    symm = []
    def add_cols(s):
        cols.extend(map(_name, s.split()))
    def add_symm(s):
        from .symmetry import parse_symmetries
        symm.extend(parse_symmetries(s))
    for line in comments:
        detect_prefix(line.strip(), '::', add_cols)
        detect_prefix(line.strip(), '>>', add_symm)
    return matrix, cols or None, symm


def read_system(filename):
    """Read linear system from file, return tuple (matrix, colnames)."""
    matrix, cols, symmetries = _read_system(filename)
    if '_' in cols:
        ccol = cols.index('_')
        if np.allclose(matrix[:,ccol], 0):
            matrix = np.delete(matrix, ccol, axis=1)
            del cols[ccol]
    return matrix, cols, symmetries


def _read_system(filename):
    if filename == '-':
        return read_system_from_file(sys.stdin)
    else:
        with open(filename) as f:
            return read_system_from_file(f)


def _unique(items):
    ret = []
    seen = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            ret.append(item)
    return ret


def _name_list(s):
    try:
        return int(s)
    except TypeError:
        return s
    except ValueError:
        pass
    if path.exists(s):
        with open(s) as f:
            return f.read().split()
    return s.split()


def _fmt_float(f):
    if round(f) == f:
        return str(int(f))
    return str(f)


def _coef(coef):
    if coef < 0:
        prefix = '-'
        coef = -coef
    else:
        prefix = '+'
    if coef != 1:
        prefix += ' ' + _fmt_float(coef)
    return prefix


def _sort_col_indices(constraint, columns):
    # len() is used as approximation for number of terms involved. For most
    # cases this should be fine.
    key = lambda i: (constraint[i] > 0, abs(constraint[i]), len(columns[i]))
    nonzero = (i for i, c in enumerate(constraint) if c != 0)
    return sorted(nonzero, key=key, reverse=True)


def format_human_readable(constraint, columns, indices=None):
    if indices is None:
        indices = _sort_col_indices(constraint, columns)
    lhs = ["{} {}".format(_coef(constraint[i]), columns[i]) for i in indices]
    if not lhs:
        lhs = ["0"]
    return "{} ≥ 0".format(" ".join(lhs).lstrip('+ '))


class System:

    """
    IO utility for systems. Keeps track of column names.
    """

    def __init__(self, matrix=None, columns=None, symmetries=None):
        self.matrix = matrix
        self.columns = columns
        self.symmetries = symmetries

    @classmethod
    def load(cls, filename=None, *, default=sys.stdin, force=True):
        if not force:
            if not filename or (filename != '-' and not path.exists(filename)):
                return cls()
        return cls(*read_system(filename))

    @property
    def dim(self):
        return self.matrix.shape[1]

    @property
    def shape(self):
        return self.matrix.shape

    def __bool__(self):
        return self.matrix is not None

    def update_symmetries(self, symmetries):
        if symmetries is not None:
            self.symmetries = symmetries
        return bool(self.symmetries)

    def symmetry_group(self):
        from .symmetry import SymmetryGroup
        return SymmetryGroup.load(self.symmetries, self.columns)

    def slice(self, columns, fill=False):
        """Return reordered system. ``fill=True`` appends missing columns."""
        indices = [self._get_column_index(c) for c in columns]
        indices = _unique(indices)
        subdim = len(indices)
        if fill:
            indices += sorted(set(range(self.dim)) - set(indices))
        if self.columns:
            columns = [self.columns[i] for i in indices]
        else:
            columns = None
        return System(self.matrix[:,indices], columns, self.symmetries), subdim

    def merge(self, other):
        if not self: return other
        if not other: return self
        assert self.columns and other.columns, \
            "Need to set column names for merge operation!"
        columns = self.columns[:]
        columns += [c for c in other.columns if c not in columns]
        col_idx = [columns.index(c) for c in other.columns]
        matrix = np.zeros((self.shape[0]+other.shape[0], len(columns)))
        matrix[:self.shape[0],:self.shape[1]] = self.matrix
        matrix[self.shape[0]:,col_idx] = other.matrix
        return self.__class__(matrix, columns)

    def _get_column_index(self, col):
        try:
            return int(col)
        except ValueError:
            pass
        col = _name(col)
        return self.columns.index(col)

    def lp(self):
        """Get the LP."""
        return Problem(self.matrix)

    def prepare_for_projection(self, subspace):
        """
        Return a tuple ``(system, subspace_dimension)`` with the subspace
        occupying the columns with the lowest indices in the returned system.
        The ``subspace`` parameter can be either of:

            - integer  — subspace dimension, using the leftmost columns
            - filename — file containing the subspace column names
            - string   — string containing the subspace column names
        """
        subspace_columns = _name_list(subspace)
        if isinstance(subspace_columns, int):
            return self, subspace_columns
        return self.slice(subspace_columns, fill=True)


class SystemFile:

    """Sink for matrix files."""

    def __init__(self, filename=None, *,
                 default=sys.stdout, append=False, columns=None,
                 symmetries=None):
        self.columns = columns
        self.file_columns = columns
        self.symm_spec = symmetries
        self._seen = VectorMemory()
        self._slice = None
        self._started = False
        self._matrix = None
        if append:
            self._read_for_append(filename)
        self._print = print_to(filename, default=default, append=append)

    def _read_for_append(self, filename):
        system = System.load(filename, force=False)
        if system.matrix:
            self._matrix = system.matrix
            self._seen.add(*system.matrix)
            self._started = True
        if system.columns:
            self.file_columns = file_columns = system.columns
            if self.columns:
                self._slice = list(map(self.columns.index, file_columns))
            else:
                self.columns = file_columns

    def __call__(self, v):
        """Output the vector ``v``."""
        if self._seen(v):
            return
        if not self._started:
            if self.file_columns:
                self._print('#::', *self.file_columns)
            if self.symm_spec:
                self._print('#>>', '; '.join(map('<>'.join, self.symm_spec)))
            self._started = True
        if self._slice:
            v = v[self._slice]
        self._print(format_vector(v))

    def pprint_symmetries(self, rows, short=False):
        from .symmetry import SymmetryGroup, group_by_symmetry
        sg = SymmetryGroup.load(self.symm_spec, self.columns)
        groups = group_by_symmetry(sg, rows)
        representatives = [g[0] for g in groups]
        if short:
            self.pprint(representatives)
        else:
            for rep in representatives:
                self._pprint_group(sg, rep)
                self._print()
        return groups

    def _pprint_group(self, sg, rep):
        indices = _sort_col_indices(rep, self.columns)
        for permutation in sg.permutations:
            inverted = permutation.inverse()
            permuted = permutation(rep)
            if self._seen(permuted):
                continue
            order = [inverted.p[i] for i in indices]
            self._print(format_human_readable(permuted, self.columns, order))

    def pprint(self, rows):
        for row in rows:
            self._print(format_human_readable(row, self.columns))


def print_to(filename=None, *default_prefix,
             append=False, default=sys.stdout):
    """
    Return a print function that prints to filename.

    If filename is left empty, prints to STDOUT.
    """
    if filename and filename != '-':
        mode = 'a' if append else 'w'
        file = open(filename, mode, buffering=1)
        return partial(print, file=file)
    elif default:
        return partial(print, *default_prefix, file=default)
    else:
        return lambda *args, **kwargs: None


def repeat(func, *args, **kwargs):
    """Call the function endlessly."""
    while True:
        yield func(*args, **kwargs)


def take(count, iterable):
    """Take count elements from iterable."""
    for i, v in zip(range(count), iterable):
        yield v


def get_bits(num):
    """Return tuple of indices corresponding to 1-bits."""
    return tuple(i for i in range(num.bit_length())
                 if num & (1 << i))


def subsets(sup):
    sup = sorted(list(sup))
    for i in range(2**len(sup)):
        yield set(sup[k] for k in get_bits(i))


def supersets(sub, world):
    sub, world = set(sub), set(world)
    return map(sub.union, subsets(world - sub))


def default_column_labels(dim):
    return ['_'] + ['_'+str(i) for i in range(1, dim)]


def _column_label(index, varnames="ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    return _name({varnames[i] for i in get_bits(index)})


def column_varname_labels(varnames):
    if isinstance(varnames, int):
        varnames = [chr(ord('A') + i) for i in range(varnames)]
    dim = 2**len(varnames)
    return ['_'] + [_column_label(i, varnames) for i in range(1, dim)]


class StatusInfo:

    def __init__(self, file=sys.stderr):
        self.file = file

    def write(self, blob):
        self.file.write(blob)
        self.file.flush()

    def cursor_up(self, num_lines=1):
        self.write("\033[" + str(num_lines) + "A")

    def clear_line(self):
        self.write("\r\033[K")

    def __call__(self, *args):
        if args:
            self.clear_line()
            self.write(" ".join(args))
        else:
            self.write("\n")


def yaml_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class _Dumper(Dumper):
        pass
    def numpy_scalar_representer(dumper, data):
        return dumper.represent_data(np.asscalar(data))
    def numpy_array_representer(dumper, data):
        return dumper.represent_data([x for x in data])
    def complex_representer(dumper, data):
        return dumper.represent_data([data.real, data.imag])
    _Dumper.add_multi_representer(np.generic, numpy_scalar_representer)
    _Dumper.add_representer(np.ndarray, numpy_array_representer)
    _Dumper.add_representer(complex, complex_representer)
    return yaml.dump(data, stream, _Dumper, **kwds)
