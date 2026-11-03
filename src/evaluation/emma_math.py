"""Bounded, target-independent EMMA value comparison (SymPy 1.14.0)."""
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import math
import re

from .schema import ContractError


@dataclass(frozen=True)
class ComparisonResult:
    correctness: bool
    error: str | None = None


@dataclass(frozen=True)
class MathNode:
    kind: str
    value: str | int | None = None
    children: tuple['MathNode', ...] = ()


class MathInputError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class MathValue:
    expression: object
    had_symbol: bool
    variables: frozenset[str]
    degree: int
    terms: int


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str


_NUMBER = re.compile(r'(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?')
_NAME = re.compile(r'[A-Za-z]+')
_BITS = 4096
_MAGNITUDE = 10**100


def require_emma_math_runtime() -> None:
    try:
        import sympy
    except ImportError as exc:
        raise ContractError('EMMA requires SymPy 1.14.0') from exc
    if getattr(sympy, '__version__', None) != '1.14.0':
        raise ContractError('EMMA requires SymPy 1.14.0')


def unwrap_math(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value.startswith('$') and value.endswith('$'):
        return value[1:-1]
    if value.startswith(r'\(') and value.endswith(r'\)'):
        return value[2:-2]
    return value


def is_nonfinite_literal(value: str) -> bool:
    return re.fullmatch(r'[+-]?(?:nan|inf|infinity|∞|\\infty)', value.strip(), re.IGNORECASE) is not None


def _unsupported():
    raise MathInputError('unsupported_expression')


def _limit():
    raise MathInputError('math_limit_exceeded')


def _nonfinite():
    raise MathInputError('non_finite_value')


def _lex(text):
    if len(text) > 512:
        _limit()
    tokens = []
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        number = _NUMBER.match(text, position)
        if number:
            token = number.group()
            mantissa, *exponent = re.split('[eE]', token)
            if sum(c.isdigit() for c in mantissa) > 64:
                _limit()
            if exponent and abs(int(exponent[0])) > 100:
                _limit()
            tokens.append(_Token('number', token))
            position = number.end()
            continue
        name = _NAME.match(text, position)
        if name:
            token = name.group()
            if token not in ('sqrt', 'pi') and len(token) != 1:
                _unsupported()
            tokens.append(_Token(token if token in ('sqrt', 'pi') else 'variable', token))
            position = name.end()
            continue
        if text[position] == '\\':
            command = re.match(r'\\[A-Za-z]+', text[position:])
            if command is None or command.group() not in (r'\pi', r'\sqrt', r'\frac', r'\cdot', r'\times'):
                _unsupported()
            token = command.group()
            kind = {r'\pi': 'pi', r'\cdot': '*', r'\times': '*'}.get(token, token)
            tokens.append(_Token(kind, token))
            position += len(token)
            continue
        if text.startswith('**', position):
            tokens.append(_Token('^', '**'))
            position += 2
            continue
        if text[position] not in '+-*/^(){}':
            _unsupported()
        tokens.append(_Token(text[position], text[position]))
        position += 1
    tokens.append(_Token('end', ''))
    return tokens


def _parse(tokens):
    """Parse only the fixed grammar; grouping depth excludes flat binary chains."""
    position = 0

    def take(kind):
        nonlocal position
        if tokens[position].kind != kind:
            _unsupported()
        token = tokens[position]
        position += 1
        return token

    def group(opening, closing, depth):
        take(opening)
        value = summation(depth + 1)
        take(closing)
        return MathNode('group', children=(value,))

    def atom(depth):
        kind = tokens[position].kind
        if kind in ('number', 'variable', 'pi'):
            return MathNode(kind, take(kind).text)
        if kind == '(':
            return group('(', ')', depth)
        if kind in ('sqrt', r'\sqrt'):
            take(kind)
            child = group('(', ')', depth) if kind == 'sqrt' else group('{', '}', depth)
            return MathNode('sqrt', children=(child,))
        if kind == r'\frac':
            take(kind)
            return MathNode('/', children=(group('{', '}', depth), group('{', '}', depth)))
        _unsupported()

    def integer_literal(depth):
        closing = None
        if tokens[position].kind in ('(', '{'):
            if depth + 1 > 16:
                _limit()
            opening = tokens[position].kind
            take(opening)
            closing = ')' if opening == '(' else '}'
        sign = 1
        if tokens[position].kind in ('+', '-'):
            sign = -1 if take(tokens[position].kind).kind == '-' else 1
        token = take('number')
        if not re.fullmatch('[0-9]+', token.text):
            _unsupported()
        exponent = sign * int(token.text)
        if closing:
            take(closing)
        if abs(exponent) > 8:
            _limit()
        return exponent

    def unary(depth):
        if depth > 16:
            _limit()
        if tokens[position].kind in ('+', '-'):
            sign = take(tokens[position].kind).kind
            return MathNode('unary' + sign, children=(unary(depth + 1),))
        node = atom(depth)
        if tokens[position].kind == '^':
            take('^')
            node = MathNode('power', integer_literal(depth), (node,))
        return node

    def product(depth):
        node = unary(depth)
        while tokens[position].kind in ('*', '/'):
            operator = take(tokens[position].kind).kind
            node = MathNode(operator, children=(node, unary(depth)))
        return node

    def summation(depth):
        if depth > 16:
            _limit()
        node = product(depth)
        while tokens[position].kind in ('+', '-'):
            operator = take(tokens[position].kind).kind
            node = MathNode(operator, children=(node, product(depth)))
        return node

    tree = summation(0)
    take('end')
    return tree


def _analyze(node):
    """Check the entire original tree before any arithmetic can cancel nodes."""
    pending = [node]
    count = 0
    while pending:
        current = pending.pop()
        if current.kind not in ('number', 'variable', 'pi', 'group'):
            count += 1
        if count > 64:
            _limit()
        pending.extend(current.children)


def _integer(value):
    if abs(value).bit_length() > _BITS:
        _limit()
    return value


def _multiply_int(left, right):
    # The lower size bound rejects definitely oversized products before multiplication.
    if left and right and abs(left).bit_length() + abs(right).bit_length() - 1 > _BITS:
        _limit()
    return _integer(left * right)


def _fraction_operation(left, right, operator):
    """Reduce cross factors before bounded integer arithmetic, never via floats."""
    if operator == '/':
        if not right:
            _nonfinite()
        right = Fraction(right.denominator, right.numerator)
        operator = '*'
    if operator == '*':
        g = math.gcd(left.numerator, right.denominator)
        h = math.gcd(right.numerator, left.denominator)
        return Fraction(_multiply_int(left.numerator // g, right.numerator // h),
                        _multiply_int(left.denominator // h, right.denominator // g))
    g = math.gcd(left.denominator, right.denominator)
    a = _multiply_int(left.numerator, right.denominator // g)
    b = _multiply_int(right.numerator, left.denominator // g)
    numerator = _integer(a + b if operator == '+' else a - b)
    # Cancel the remaining common denominator factor before multiplication.
    h = math.gcd(numerator, g)
    return Fraction(numerator // h,
                    _multiply_int(left.denominator // g, right.denominator // h))


def _fraction_power(value, exponent):
    if exponent < 0:
        if not value:
            _nonfinite()
        value = Fraction(value.denominator, value.numerator)
        exponent = -exponent
    def bounded_power(integer):
        if integer and (abs(integer).bit_length() - 1) * exponent + 1 > _BITS:
            _limit()
        # At most 8 factors; no integer outside the component bound is retained.
        result = 1
        for _ in range(exponent):
            result = _multiply_int(result, integer)
        return result
    return Fraction(bounded_power(value.numerator), bounded_power(value.denominator))


def _check_numeric(expression):
    import sympy
    if expression.is_real is not True or expression.is_finite is not True:
        _nonfinite()
    for component in expression.atoms(sympy.Rational):
        _integer(int(component.p))
        _integer(int(component.q))
    if abs(expression) > _MAGNITUDE:
        _limit()


def _guard_numeric_operation(left, right, operator):
    """Project only coefficients combined by this Add/Mul/division operation."""
    def coefficient(expression):
        number, remainder = expression.as_coeff_Mul()
        return Fraction(int(number.p), int(number.q)), remainder

    if operator in ('+', '-'):
        coefficients = {}
        for expression, sign in ((left, 1), (right, 1 if operator == '+' else -1)):
            for term in expression.args if expression.is_Add else (expression,):
                number, remainder = coefficient(term)
                coefficients[remainder] = _fraction_operation(
                    coefficients.get(remainder, Fraction(0)), sign * number, '+')
        return
    a, _ = coefficient(left)
    b, _ = coefficient(right)
    _fraction_operation(a, b, operator)
    # SymPy distributes a rational scalar across an Add automatically. Other
    # products retain the Add as a factor, so its unrelated terms stay separate.
    if operator == '*' and left.is_Rational and right.is_Add:
        for term in right.args:
            _fraction_operation(a, coefficient(term)[0], '*')
    elif left.is_Add and right.is_Rational:
        for term in left.args:
            _fraction_operation(coefficient(term)[0], b, operator)


def _polynomial_operation(left, right, operator, exponent=None):
    """Bound every rational coefficient before constructing an expanded expression."""
    import sympy
    symbols = sorted(left.free_symbols | (right.free_symbols if right is not None else set()), key=str)
    def coefficients(expression):
        if not expression.free_symbols:
            return {(0,) * len(symbols): Fraction(int(expression.p), int(expression.q))}
        polynomial = sympy.Poly(expression, *symbols, domain='QQ')
        return {powers: Fraction(int(coef.p), int(coef.q)) for powers, coef in polynomial.terms()}
    def multiply(a, b):
        result = {}
        for powers_a, coef_a in a.items():
            for powers_b, coef_b in b.items():
                powers = tuple(x + y for x, y in zip(powers_a, powers_b))
                term = _fraction_operation(coef_a, coef_b, '*')
                result[powers] = _fraction_operation(result.get(powers, Fraction(0)), term, '+')
        return result
    a = coefficients(left)
    if operator == 'power':
        result = {(0,) * len(symbols): Fraction(1)}
        for _ in range(exponent):
            result = multiply(result, a)
    elif operator == '/':
        divisor = Fraction(int(right.p), int(right.q))
        result = {powers: _fraction_operation(coef, divisor, '/') for powers, coef in a.items()}
    else:
        b = coefficients(right)
        if operator == '*':
            result = multiply(a, b)
        else:
            result = dict(a)
            for powers, coef in b.items():
                result[powers] = _fraction_operation(result.get(powers, Fraction(0)), coef, operator)
    terms = []
    for powers, coef in result.items():
        if abs(coef) > _MAGNITUDE:
            _limit()
        terms.append(sympy.Mul(sympy.Rational(coef.numerator, coef.denominator),
                               *(sympy.Pow(symbol, power) for symbol, power in zip(symbols, powers) if power)))
    return sympy.Add(*terms)


def _evaluate(node):
    import sympy
    if node.kind == 'number':
        number = Fraction(Decimal(node.value))
        expression = sympy.Rational(number.numerator, number.denominator)
        _check_numeric(expression)
        return MathValue(expression, False, frozenset(), 0, 1)
    if node.kind == 'pi':
        return MathValue(sympy.pi, False, frozenset(), 0, 1)
    if node.kind == 'variable':
        return MathValue(sympy.Symbol(node.value), True, frozenset((node.value,)), 1, 1)
    values = tuple(_evaluate(child) for child in node.children)
    a = values[0]
    if node.kind == 'group':
        return a
    variables = frozenset().union(*(value.variables for value in values))
    had_symbol = any(value.had_symbol for value in values)
    degree, terms = a.degree, a.terms
    if node.kind in ('+', '-'):
        degree = max(value.degree for value in values)
        terms = sum(value.terms for value in values)
    elif node.kind == '*':
        degree = sum(value.degree for value in values)
        terms = math.prod(value.terms for value in values)
    elif node.kind == 'power':
        if had_symbol and node.value < 0:
            _unsupported()
        degree *= max(0, node.value)
        terms **= abs(node.value)
    if len(variables) > 8 or degree > 8 or terms > 128:
        _limit()
    if node.kind == 'sqrt' and had_symbol:
        _unsupported()
    if had_symbol:
        if any(not value.expression.free_symbols and not value.expression.is_Rational for value in values):
            _unsupported()
        if node.kind == '/' and values[1].had_symbol:
            _unsupported()
    if node.kind == '/' and values[1].expression == 0:
        _nonfinite()
    if node.kind.startswith('unary'):
        expression = a.expression if node.kind == 'unary+' else -a.expression
    elif node.kind == 'sqrt':
        if a.expression < 0:
            _nonfinite()
        expression = sympy.sqrt(a.expression)
    elif had_symbol:
        expression = _polynomial_operation(a.expression, values[1].expression if len(values) == 2 else None,
                                           node.kind, node.value)
    elif node.kind == 'power':
        for component in a.expression.atoms(sympy.Rational):
            _fraction_power(Fraction(int(component.p), int(component.q)), node.value)
        expression = sympy.Pow(a.expression, node.value)
    else:
        b = values[1]
        if a.expression.is_Rational and b.expression.is_Rational:
            value = _fraction_operation(Fraction(int(a.expression.p), int(a.expression.q)),
                                        Fraction(int(b.expression.p), int(b.expression.q)), node.kind)
            expression = sympy.Rational(value.numerator, value.denominator)
        else:
            _guard_numeric_operation(a.expression, b.expression, node.kind)
            if node.kind == '+':
                expression = sympy.Add(a.expression, b.expression)
            elif node.kind == '-':
                expression = sympy.Add(a.expression, -b.expression)
            elif node.kind == '*':
                expression = sympy.Mul(a.expression, b.expression)
            else:
                expression = sympy.Mul(a.expression, sympy.Pow(b.expression, -1))
    if not expression.free_symbols:
        _check_numeric(expression)
    return MathValue(expression, had_symbol, variables, degree, terms)


def compare_values(left: str, right: str) -> ComparisonResult:
    require_emma_math_runtime()
    import sympy
    left, right = unwrap_math(left), unwrap_math(right)
    if is_nonfinite_literal(left) or is_nonfinite_literal(right):
        return ComparisonResult(False, 'non_finite_value')
    if not left.strip() or not right.strip():
        return ComparisonResult(False, 'unsupported_expression')
    if left == right:
        return ComparisonResult(True)
    try:
        a_tree, b_tree = _parse(_lex(left)), _parse(_lex(right))
        _analyze(a_tree)
        _analyze(b_tree)
        a, b = _evaluate(a_tree), _evaluate(b_tree)
        if (a.had_symbol or b.had_symbol) and any(
            not value.expression.free_symbols and not value.expression.is_Rational for value in (a, b)
        ):
            _unsupported()
        if not a.expression.free_symbols and not b.expression.free_symbols:
            return ComparisonResult(round(float(a.expression.evalf()), 2) == round(float(b.expression.evalf()), 2))
        symbols = sorted(a.expression.free_symbols | b.expression.free_symbols, key=str)
        # Compare admitted QQ coefficients without constructing an unbounded difference.
        return ComparisonResult(sympy.Poly(a.expression, *symbols, domain='QQ') ==
                                sympy.Poly(b.expression, *symbols, domain='QQ'))
    except MathInputError as exc:
        return ComparisonResult(False, exc.code)
