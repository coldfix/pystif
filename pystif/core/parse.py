"""
Utilities for parsing input files with systems of linear (in-)equalities.

The input file grammar looks somewhat like this:

    line        ::=     statement? comment?
    statement   ::=     equation | var_decl | markov | mutual
    comment     ::=     r"#.*"

    equation    ::=     expression relation expression
    relation    ::=     ">=" | "≥" | "<=" | "≤" | "="
    expression  ::=     sign? term (sign term)*
    term        ::=     (number "*"?)? symbol | number

    markov      ::=     "markov" var_list (":" var_list){2+}  ("|" var_list)?
    mutual      ::=     "mutual" var_list (":" var_list)+     ("|" var_list)?

    var_decl    ::=     "rvar" var_list
    var_list    ::=     (identifier ","?)*

    symbol      ::=     identifier | entropy | mutual_info
    entropy     ::=     "H(" var_list ("|" var_list)? ")"
    mutual_info ::=     "I(" var_list (":" var_list)+ ("|" var_list)? )"

For example:

    3 X + 4Y = 0
      X + 2  >= 4*Y

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
# - [PRO] BETTER ERROR MESSAGES
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

from functools import partial

import numpy as np
import funcparserlib.lexer as fpll
import funcparserlib.parser as fplp

from .io import column_varname_labels
from .it import elemental_inequalities


def make_lexer():
    """
    Return a tokenizer for the input file grammar.

    () -> (str -> [Token])
    """
    tokenizer = fpll.make_tokenizer([
        ('COMMENT',     (r'#.*',)),
        ('WS',          (r'[ \t]+',)),
        ('NAME',        (r'[a-zA-Z_]\w*',)),
        ('NUMBER',      (r'[-+]?\d+(\.\d+)?([eE][+\-]?\d+)?',)),
        ('OP',          (r'[-+*,:;()≤=≥|]|>=|<=',)),
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

        ParseResult = [ParseIneq].
        ParseIneq = [(str, float)]

    In other words, the parser returns a list of inequalities that are
    each represented by a list of variable names and coefficients. There can
    be multiple terms for the same variable in the same inequality.
    """
    # TODO:
    # - variable:   information_measure | identifier
    # - line:       equation | elemental inequalities | positivities | q space

    from funcparserlib.parser import maybe, skip, finished, pure

    def many(p, min=0):
        if min == 0:
            return fplp.many(p)
        q = p + many(p, min-1) >> collapse
        return q.named('(%s , { %s })' % (p.name, p.name))

    v = pure
    X = partial(exact, 'OP')
    L = lambda text: skip(literal(text))
    Lo = lambda text: skip(maybe(literal(text)))

    # primitives
    sign        = X('+') | X('-')
    relation    = X('>=') | X('<=') | X('≥') | X('≤') | X('=')
    number      = some('NUMBER')                        >> tokval >> to_number
    identifier  = some('NAME')                          >> tokval
    variable    = identifier                            >> make_variable
    colon       = L(':') | L(';')

    def var_g(n):
        # n: minimum number of parts
        if n == 0:
            return var_g(1) | v([])
        return var_list + many(colon + var_list, n-1)   >> collapse

    # information measures
    var_list    = many(identifier + Lo(','))            >> set
    conditional = L('|') + var_list | v(set())
    entropy     = L('H(') + var_list + conditional + L(')')     >> make_entropy
    mutual_info = L('I(') + var_g(2) + conditional + L(')')     >> make_mut_inf

    # (in-)equalities
    symbol      = entropy | mutual_info | variable
    var_term    = (number + Lo('*') | v(1)) + symbol    >> make_var_term
    constant    = number                                >> make_constant
    term        = var_term | constant
    f_term      = (sign | v('+')) + term                >> make_term
    s_term      = sign + term                           >> make_term
    expression  = f_term + many(s_term)                 >> collapse >> make_expr
    equation    = expression + relation + expression    >> make_equation

    # commands
    var_decl    = L('rvar') + var_list                  >> make_random_vars
    markov      = L('markov') + var_g(3) + conditional  >> make_markov_chain
    mutual      = L('mutual') + var_g(2) + conditional  >> make_mutual_indep

    # toplevel
    eof         = skip(finished)
    line        = (equation | var_decl | markov | mutual | v([])) + eof

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


def to_parse_inequality(numpy_vector, colnames):
    """
    np.array, [str] -> ParseIneq = [(str, float)]
    """
    return [
        (colnames[idx], coef)
        for idx, coef in enumerate(numpy_vector)
        if coef != 0
    ]


def to_parse_result(vectors, colnames):
    """
    Back-transform a numpy array to a parse result.

    np.array, [str] -> ParseResult
    """
    return [to_parse_inequality(v, colnames) for v in vectors]


#----------------------------------------
# Parser internals
#----------------------------------------

def tokval(tok):
    return tok.value


def exact(token_type, text):
    return fplp.a(fpll.Token(token_type, text)) >> tokval


def const(value):
    return lambda tok: value


def stararg(func):
    return lambda args: func(*args)


def literal(text):
    """
    Matches any text that will produce the same series of filtered tokens as
    the given text, i.e. the text plus optional whitespaces between tokens.
    """
    tokenize = make_lexer()
    tokens = map(fplp.a, tokenize(text))
    return sum(tokens, next(tokens)) >> const(text)


def tt(*token_type):
    return lambda t: t.type in token_type


def some(*token_type):
    return fplp.some(tt(*token_type))


#----------------------------------------
# transformation functions for result
#----------------------------------------

def to_number(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def make_term(signed_term):
    sign, terms = signed_term
    sign = 1 if sign == '+' else -1
    return [(name, sign*coef) for name, coef in terms]


def make_expr(expr):
    return [term for subexpr in expr for term in subexpr]


def collapse(items):
    return [items[0]] + items[1]


@stararg
def make_equation(lhs, rel, rhs):
    negate = lambda terms: [(name, -coef) for name, coef in terms]
    pos = lhs + negate(rhs)
    neg = rhs + negate(lhs)
    if rel == '>=' or rel == '≥':
        return [pos]
    if rel == '<=' or rel == '≤':
        return [neg]
    return [pos, neg]


def make_random_vars(varnames):
    """
    Iterate all elemental inequalities for the given variables.

    An inequality is represented as a list of pairs (name, coef).

    [str] -> ParseResult
    """
    varnames = sorted(varnames)
    colnames = column_varname_labels(varnames)
    num_vars = len(varnames)
    return to_parse_result(elemental_inequalities(num_vars), colnames)


def _entropy_colname(var_list):
    return "".join(sorted(var_list))


def make_constant(num):
    return [('_', num)]


@stararg
def make_var_term(coef, terms):
    return [(n, coef*c) for n, c in terms]


def make_variable(varname):
    return [(varname, 1)]


@stararg
def make_entropy(core, cond):
    if cond:
        return [
            (_entropy_colname(core|cond), 1),
            (_entropy_colname(cond), -1),
        ]
    return [(_entropy_colname(core), 1)]


def result_to_list(func):
    return lambda *args, **kwargs: list(func(*args, **kwargs))


@stararg
@result_to_list
def make_mut_inf(parts, cond):
    # Multivariate mutual information is recursively defined by
    #
    #          I(a:…:y:z) = I(a:…:y) - I(a:…:y|z)
    #
    # Here, it is calculated as the alternating sum of (conditional)
    # entropies of all subsets of the parts [Jakulin & Bratko (2003)].
    #
    # See: http://en.wikipedia.org/wiki/Multivariate_mutual_information
    num_parts = len(parts)
    num_subsets = 2 ** num_parts

    # Start at i=1 because i=0 which corresponds to H(empty set) gives no
    # contribution to the sum. Furthermore, the i=0 is already reserved
    # for the constant term for our purposes.
    for i in range(1, num_subsets):
        core = set()
        sign = -1
        for j, part in enumerate(parts):
            if i & (1<<j):
                core |= part;
                sign = -sign
        yield (_entropy_colname(core|cond), sign)

    # The coefficient of the conditional part always sums to one: There are
    # equally many N bit strings with odd/even number of set bits. Since we
    # ignore zero, there is one more "odd string".
    if cond:
        yield (_entropy_colname(cond), 1)


@stararg
@result_to_list
def make_markov_chain(parts, cond):
    A = set()
    for a, b, c in zip(parts[:-2], parts[1:-1], parts[2:]):
        A |= a
        yield from make_mutual_indep(([A, c], b|cond))


@stararg
def make_mutual_indep(parts, cond):
    # H(a,b,c,…|z) = H(a|z) + H(b|z) + H(c|z) + …
    lhs = [(_entropy_colname(set.union(cond, *parts)), 1)]
    rhs = [(_entropy_colname(set.union(cond, part)), 1) for part in parts]
    if cond:
        lhs += [(_entropy_colname(cond), -1)]
        rhs += [(_entropy_colname(cond), -len(parts))]
    return make_equation((lhs, '=', rhs))


#----------------------------------------
# ad-hoc testing:
#----------------------------------------

if __name__ == '__main__':
    d = parse_file([
        'H(X,Y, Z | X,Z) >= 0',
        'I(X,Y: Z | D) >= 0',
        'rvar x y',
        '- hello + 2 x + x ≤ world+ 3',
        '- x + x + x >= 2 x + 3',
    ])
    print(d)
    v, n = to_numpy_array(d)
    print(n)
    print(v)
