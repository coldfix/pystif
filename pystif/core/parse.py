"""
Utilities for parsing input files with systems of linear (in-)equalities.

The input file grammar looks somewhat like this:

    line        ::=     equation? comment?
    comment     ::=     r"#.*"
    equation    ::=     terms relation terms
    relation    ::=     ">=" | "≥" | "<=" | "≤" | "="
    terms       ::=     sign? term (sign term)*
    term        ::=     number? name | number

For example:

    3 X + 4Y = 0
      X + 2  >= 3

The parser can be invoked using the ``parse_file`` or ``parse_files``
functions. The results of multiple invocations can be combined using the
``merge_parse_results`` function. The final parse result can then be converted
to a numpy array and a list of column names using the ``to_numpy_array``
function.
"""

__all__ = [
    'parse_file',
    'parse_files',
    'merge_parse_results',
    'to_numpy_array',
]

# pyparsing:
# - [PRO] automatic whitespace removal
# - [PRO] set parser action on the object (=> can be set later easily)
# - [CON] no tokenizer
# - [CON] awkward argument/return type of parser action callback
# - [CON] more verbose
#
# tokenize:
# - [PRO] well established tokens, good number recognition
# - [CON] no custom tokens (≤,≥)
# - [CON] have to write own Token class

import numpy as np
import funcparserlib.lexer as fpll
import funcparserlib.parser as fplp


def make_lexer():
    """
    Return a tokenizer for the input file grammar.

    () -> (str -> [Token])
    """
    tokenizer = fpll.make_tokenizer([
        ('COMMENT',     (r'#.*',)),
        ('WS',          (r'[ \t]+',)),
        ('NAME',        (r'[a-zA-Z_]\w*',)),
        ('NUMBER',      (r'[+\-]?\d+(\.\d+)?([eE][+\-]?\d+)?',)),
        ('OP',          (r'[+\-]|≥|>=|≤|<=|=',)),
    ])
    def tokenize(text):
        trash = tt('COMMENT', 'WS')
        return [tok for tok in tokenizer(text) if not trash(tok)]
    return tokenize


def make_parser():
    """
    Return a parser that parses a system of linear (in-)equalities.

    () -> ([Token] -> ParseResult)

    where the exact type of ParseResult is an implementation detail. The
    current form is

        ParseResult = [Inequality].
        Inequality = [(str, float)]

    In other words, the parser returns a list of inequalities that are
    each represented by a list of variable names and coefficients. There can
    be multiple terms for the same variable in the same inequality.
    """
    # TODO:
    # - variable:   information_measure | identifier
    # - line:       equation | elemental inequalities | positivities | q space

    v = fplp.pure

    # primitives
    sign        = op('+') | op('-')
    relation    = op('>=') | op('<=') | op('≥') | op('≤') | op('=')
    number      = some('NUMBER')                        >> tokval >> to_number
    identifier  = some('NAME')                          >> tokval
    variable    = identifier
    var_term    = (number | v(1)) + variable
    constant    = number + v('_')

    # (in-)equalities
    term        = var_term | constant
    f_term      = (sign | v('+')) + term                >> process_term
    s_term      = sign + term                           >> process_term
    expression  = f_term + fplp.many(s_term)            >> collapse
    equation    = expression + relation + expression    >> make_equation

    # toplevel
    eof         = fplp.skip(fplp.finished)
    line        = (equation | v([])) + eof
    return line


def merge_parse_results(results):
    """
    Combine the results of multiple parser runs.

    [ParseResult] -> ParseResult
    """
    return sum(results, [])


def parse_file(lines):
    """
    Parses a system of linear (in-)equalities from one file. The parameter
    must be given as an iterable of lines.

    [str] -> ParseResult
    """
    tokenize = make_lexer()
    parser = make_parser()
    return merge_parse_results(
        parser.parse(tokenize(l)) for l in lines)


def _lines(filename):
    """
    Iterate over all lines in the file. If the parameter is not an existing
    file name, treat it as the file content itself.

    (Filename | Text) -> [str]
    """
    try:
        with open(filename) as f:
            yield from f
    except FileNotFoundError:
        yield from filename.split('\n')


def parse_files(files):
    """
    [Filename | Text] -> ParseResult
    """
    return merge_parse_results(
        parse_file(_lines(f)) for f in files)


#----------------------------------------
# Finalization of parse results
#----------------------------------------

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


def to_numpy_array(parse_result, col_names=('_',)):
    """
    Further transform parser output to numpy array, also return list of column
    names in order.

    ParseResult -> np.array, [str]
    """
    col_idx = AutoInsert(list(col_names))
    indexed = [[(col_idx[name], coef)
                for name, coef in terms]
               for terms in parse_result]
    result = np.zeros((len(indexed), len(col_idx)))
    for row, terms in enumerate(indexed):
        for idx, coef in terms:
            result[row][idx] += coef
    return result, col_idx._cols


#----------------------------------------
# Parser internals
#----------------------------------------

def tokval(tok):
    return tok.value


def op(text):
    return fplp.a(fpll.Token('OP', text)) >> tokval


def tt(*token_type):
    return lambda t: t.type in token_type


def some(*token_type):
    return fplp.some(tt(*token_type))


def to_number(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def process_term(signed_term):
    sign, (coef, name) = signed_term
    sign = 1 if sign == '+' else -1
    return name, sign*coef


def collapse(items):
    return [items[0]] + items[1]


def make_equation(equation):
    lhs, rel, rhs = equation
    negate = lambda terms: [(name, -coef) for name, coef in terms]
    pos = lhs + negate(rhs)
    neg = rhs + negate(lhs)
    if rel == '>=' or rel == '≥':
        return [pos]
    if rel == '<=' or rel == '≤':
        return [neg]
    return [pos, neg]


#----------------------------------------
# ad-hoc testing:
#----------------------------------------

if __name__ == '__main__':
    d = parse_file([
        '- hello + 2 x + x ≤ world+ 3',
        '- x + x + x >= 2 x + 3',
    ])
    print(d)
    v, n = to_numpy_array(d)
    print(n)
    print(v)
