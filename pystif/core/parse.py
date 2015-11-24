


def create_index(l):
    return {v: i for i, v in enumerate(l)}


class AutoInsert:

    """Autovivification for column names."""

    def __init__(self, cols):
        self._cols = cols
        self._idx = create_index(cols)

    def __len__(self):
        return len(self._cols)

    def __getitem__(self, key):
        try:
            return self._idx[key]
        except KeyError:
            self._cols.append(key)
            self._idx[key] = index = len(self._cols) - 1
            return index


def _parse_expr(expr):
    re_number = r'\s*([+\-]?(?:\d+(?:\.\d*)?|\d*\.\d+)(?:[eE][+\-]?\d+)?)'
    re_ident = r'\s*([a-zA-Z_][\w\.]*)'
    re_sign = r'\s*([-+])'
    re_head = re.compile("".join((
        '^', re_sign, '?', re_number, '?', re_ident,
        '|^', re_sign, '?', re_number
    )))
    re_tail = re.compile("".join((
        '^', re_sign, re_number, '?', re_ident,
        '|^', re_sign, re_number
    )))
    signs = {'+': 1, '-': -1, None: 1}
    m = re_head.match(expr)
    while m:
        ident = m.group(3)
        if ident:
            sign, number = m.groups()[:2]
        else:
            ident = 0
            sign, number = m.groups()[3:]
        coef = signs[sign]
        if number is not None:
            coef *= float(number)
        yield (ident, coef)
        expr = expr[m.end():]
        m = re_tail.match(expr)
    if expr.strip():
        raise ValueError("Unexpected token at: {!r}".format(expr))


def _parse_eq_line(line, col_idx):
    line = line.strip()
    if not line or line.startswith('#'):
        return []
    m = re.match("^([^≥≤<>=]*)(≤|≥|<=|>=|=)([^≥≤<>=]*)$", line)
    if not m:
        raise ValueError("Invalid constraint format: {!r}.\n"
                         "Must contain exactly one relation".format(line))
    lhs, rel, rhs = m.groups()
    terms = list(_parse_expr(lhs))
    terms += [(col, -coef) for col, coef in _parse_expr(rhs)]
    indexed = [(0 if col == 0 else col_idx[col], coef)
               for col, coef in terms]
    v = np.zeros(len(col_idx))
    for idx, coef in indexed:
        v[idx] += coef
    if rel == '<=' or rel == '≤':
        return [-v]
    if rel == '>=' or rel == '≥':
        return [v]
    return [-v, v]


def _parse_eq_file(eq_str, col_idx):
    if not path.exists(eq_str):
        return _parse_eq_line(eq_str, col_idx)
    with open(eq_str) as f:
        return sum((_parse_eq_line(l, col_idx) for l in f), [])
