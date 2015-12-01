r"""
Utilities for parsing input files with systems of linear (in-)equalities.

The input file grammar looks somewhat like this:

    document    ::=     line ("\n" line)*
    line        ::=     statement? comment?
    statement   ::=     equation | var_decl | markov | mutual
    comment     ::=     r"#.*"

    equation    ::=     expression relation expression
    relation    ::=     ">=" | "≥" | "<=" | "≤" | "==" | "="
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

The parser can be invoked using the ``parse_text`` or ``parse_files``
functions. The final parse result can then be converted to a numpy array and
a list of column names using the ``to_numpy_array`` function.
"""


# TODO:
# - support multi-char variables‽
# - raise exception for multi-char variables
# - explicit column aliases ("alias" statement)
# - implicit column aliases (when defining information inequalities)
# - "symmetry" statement
# - result of statements should be callback objects
# - composed expressions
# - complement operator for var lists '~'
# - unit tests for this module
# - more doc strings
#
# - keep input statements as comments for *makesys*
# - recognize and output "rvar" statement for *pretty*


__all__ = [
    'parse_text',
    'parse_files',
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
from itertools import chain

import numpy as np
import funcparserlib.lexer as fpll
import funcparserlib.parser as fplp
from funcparserlib.parser import maybe, skip, finished, pure

from .io import column_varname_labels
from .it import elemental_inequalities
from .vector import Vector, _name


class Statement:

    """
    Statements (a.k.a. definitions/commands) are the building blocks of the
    parser output. Each statement can define a bunch of properties that must
    be part of the output system.
    """

    def columns(self) -> [str]:
        """Return list of columns required/defined by this statement."""
        return ()

    def constraints(self, col_idx: {str: int}) -> np.array:
        """Return constraint matrix defined by this statement."""
        return np.empty((0, len(col_idx)))


def tokenize(text: str) -> [fpll.Token]:
    """Break an input string into a list of tokens accepted by the parser."""
    trash = tt('COMMENT', 'WS')
    return [tok for tok in _tokenizer(text) if not trash(tok)]


def parse_text(text: str) -> [Statement]:
    """Parse a system of linear (in-)equalities from one file."""
    return document.parse(tokenize(text))


def _content(filename: str) -> str:
    """
    Get the file content as a blob of text. If the parameter is not the name
    of an existing file, treat it as the file content itself.
    """
    try:
        with open(filename) as f:
            return f.read()
    except FileNotFoundError:
        return filename


def parse_files(files: [str]) -> [Statement]:
    """Concat and parse multiple files/expressions as one system."""
    return parse_text(
        "\n".join(map(_content, files)))


#----------------------------------------
# Finalization of parse results
#----------------------------------------

def create_index(l: list) -> dict:
    """Create an index of the list items' indices."""
    return {v: i for i, v in enumerate(l)}


def list_from_index(idx: {str: int}) -> [str]:
    """Inverse of the create_index() function."""
    return sorted(idx, key=lambda k: idx[k])


def to_numpy_array(parse_result: [Statement]) -> (np.array, [str]):
    """
    Further transform parser output to numpy array, also return list of column
    names in order.
    """
    col_idx = {}
    for stmt in statements:
        for col in stmt.columns():
            col_idx.setdefault(_name(col), len(col_idx))
    result = np.vstack(stmt.constraints(col_idx) for stmt in statements)
    return result, list_from_index(col_idx)


#----------------------------------------
# Utility functions
#----------------------------------------

def stararg(func: "(P…), -> R") -> "P… -> R":
    """Unpack argument before invoking function."""
    return lambda args: func(*args)


def returns(result_type: "R -> R'") -> "(P -> R) -> (P -> R')":
    """Cast function result to result_type."""
    def decorate(func: "P -> R") -> "P -> R'":
        return lambda *args, **kwargs: result_type(func(*args, **kwargs))
    return decorate


def flatten(ll: [[any]]) -> [any]:
    """Flatten a two-level nested list."""
    return list(chain.from_iterable(ll))


def unslice(slice: np.array, indices: [int], num_cols: int) -> np.array:
    slice = np.asarray(slice)
    res = np.zeros((slice.shape[0], num_cols))
    res[:,indices] = slice
    return res


#----------------------------------------
# Parser internals
#----------------------------------------

def many(p, min=0):
    if min == 0:
        return fplp.many(p)
    q = p + many(p, min-1) >> collapse
    return q.named('(%s , { %s })' % (p.name, p.name))


def tokval(tok):
    return tok.value


def exact(token_type, text):
    return fplp.a(fpll.Token(token_type, text)) >> tokval


def const(value):
    return lambda tok: value


def literal(text):
    """
    Matches any text that will produce the same series of filtered tokens as
    the given text, i.e. the text plus optional whitespaces between tokens.
    """
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


def make_expr(expr):
    return sum(expr, Vector())


def collapse(items):
    return [items[0]] + items[1]


class SimpleConstraintList(Statement):

    def __init__(self, rows: [Vector]):
        self.rows = list(rows)

    def columns(self):
        return flatten(self.rows)

    def constraints(self, col_idx):
        res = np.zeros((len(self.rows), len(col_idx)))
        for i, r in enumerate(self.rows):
            for k, v in r.items():
                res[i][col_idx[k]] = v
        return res


@returns(SimpleConstraintList)
def Constraint(lhs, rel, rhs):
    if rel == '>=' or rel == '≥':
        return [lhs - rhs]
    if rel == '<=' or rel == '≤':
        return [rhs - lhs]
    return [lhs - rhs, rhs - lhs]


class VarDecl(Statement):

    """
    Iterate all elemental inequalities for the given variables.

    An inequality is represented as a list of pairs (name, coef).
    """

    def __init__(self, varnames: [str]):
        self.varnames = sorted(varnames)
        self.num_vars = len(varnames)

    def columns(self):
        return column_varname_labels(self.varnames)

    def constraints(self, col_idx):
        return unslice(list(elemental_inequalities(self.num_vars)),
                       [col_idx[c] for c in self.columns()],
                       len(col_idx))


@stararg
def scale_vector(coef, vector):
    return vector * coef


def make_variable(varname):
    return Vector({varname: 1})


def make_constant(num):
    return Vector({'_': num})


@stararg
def make_entropy(core, cond):
    return Vector((
        (core|cond, 1),
        (cond, -1),
    ))


@stararg
@returns(Vector)
def make_mut_inf(parts: "[VarList]", cond: "VarList") -> Vector:
    # Multivariate mutual information is recursively defined by
    #
    #          I(a:…:y:z) = I(a:…:y) - I(a:…:y|z)
    #
    # Here, it is calculated as the alternating sum of (conditional)
    # entropies of all subsets of the parts [Jakulin & Bratko (2003)].
    #
    # See: http://en.wikipedia.org/wiki/Multivariate_mutual_information
    num_subsets = 2 ** len(parts)

    # Start at i=1 because i=0 which corresponds to H(empty set) gives no
    # contribution to the sum. Furthermore, the i=0 is already reserved
    # for the constant term for our purposes.
    for i in range(1, num_subsets):
        subs = [part for j, part in enumerate(parts) if i & (1<<j)]
        core = set.union(*subs)
        sign = -(-1)**len(subs)
        yield (core|cond, sign)

    # The coefficient of the conditional part always sums to one: There are
    # equally many N bit strings with odd/even number of set bits. Since we
    # ignore zero, there is one more "odd string".
    if cond:
        yield (cond, -1)


@returns(SimpleConstraintList)
def MarkovChain(parts: "[VarList]", cond: "VarList") -> Statement:
    A = set()
    for a, b, c in zip(parts[:-2], parts[1:-1], parts[2:]):
        A |= a
        yield from MutualIndep(([A, c], b|cond)).rows


def MutualIndep(parts: "[VarList]", cond: "VarList") -> Statement:
    # H(a,b,c,…|z) = H(a|z) + H(b|z) + H(c|z) + …
    lhs = [(set.union(cond, *parts), 1)]
    rhs = [(set.union(cond, part), 1) for part in parts]
    if cond:
        lhs += [(cond, -1)]
        rhs += [(cond, -len(parts))]
    return Constraint(Vector(lhs), '=', Vector(rhs))


#----------------------------------------
# lexer/parser implementation
#----------------------------------------

_tokenizer = fpll.make_tokenizer([
    ('NEWLINE',     (r'[\r\n]+',)),
    ('COMMENT',     (r'#.*',)),
    ('WS',          (r'[ \t]+',)),
    ('NAME',        (r'[a-zA-Z_]\w*',)),
    ('NUMBER',      (r'[-+]?\d+(\.\d+)?([eE][+\-]?\d+)?',)),
    ('OP',          (r'[<=>]=|[-+*,:;()≤=≥|]',)),
])

# TODO:
# - variable:   information_measure | identifier
# - line:       equation | elemental inequalities | positivities | q space

v = pure
X = partial(exact, 'OP')
L = lambda text: skip(literal(text))
Lo = lambda text: skip(maybe(literal(text)))

# primitives
sign        = X('+') >> const(+1) | X('-') >> const(-1)
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
var_term    = (number + Lo('*') | v(1)) + symbol    >> scale_vector
constant    = number                                >> make_constant
term        = var_term | constant
f_term      = (sign | v(+1)) + term                 >> scale_vector
s_term      = sign + term                           >> scale_vector
expression  = f_term + many(s_term)                 >> collapse >> make_expr
equation    = expression + relation + expression    >> stararg(Constraint)

# commands
var_decl    = L('rvar') + var_list                  >> VarDecl
markov      = L('markov') + var_g(3) + conditional  >> stararg(MarkovChain)
mutual      = L('mutual') + var_g(2) + conditional  >> stararg(MutualIndep)
empty       = v(Statement())

# toplevel
eol         = skip(some('NEWLINE'))
eof         = skip(finished)
line        = equation | var_decl | markov | mutual | empty
document    = line + many(eol + line) + eof         >> collapse


#----------------------------------------
# ad-hoc testing:
#----------------------------------------

if __name__ == '__main__':
    d = parse_text("""
        H(X,Y, Z | X,Z) >= 0
        I(X,Y: Z | D) >= 0
        rvar x y
        - hello + 2 x + x ≤ world+ 3
        - x + x + x >= 2 x + 3
    """)
    print(d)
    v, n = to_numpy_array(d)
    print(n)
    print(v)
