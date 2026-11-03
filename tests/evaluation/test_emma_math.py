"""CPU contract witnesses for the deliberately bounded EMMA comparator."""
import builtins
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('left,right,correct', [
    ('0.5', '1/2', True), (r'\frac{1}{2}', '0.5', True),
    ('sqrt(4)', '2', True), (r'\sqrt{4}', '2', True),
    ('pi', '3.14', True), ('1.5e-3', '0', True),
    ('(1e-100)^2', '0', True), ('1.004', '1', True),
    ('1.006', '1', False), ('2.675', '2.67', True),
    ('x+x', '2*x', True), ('x+x', '3*x', False),
    ('(x+1)^2', 'x^2+2*x+1', True), ('X', 'x', False),
    ('-2^2', '-4', True), ('(2^3)^2', '64', True),
    ('sin(x)', 'sin(x)', True), ('a b', 'ab', False),
    ('x-x', '0', True), ('x/2', '0.5*x', True),
    (r'2\cdot x', 'x+x', True), (r'2\times x', 'x+x', True),
    (r'\pi', '3.14', True), ('2^-2', '.25', True),
    ('2^{+2}', '4', True), ('i+i', '2*i', True),
])
def test_math_contract(left, right, correct):
    from evaluation.emma_math import compare_values
    assert compare_values(left, right).correctness is correct
    assert compare_values(right, left).correctness is correct


@pytest.mark.parametrize('value,code', [
    ('2x', 'unsupported_expression'), ('x/x', 'unsupported_expression'),
    ('sqrt(x)', 'unsupported_expression'), ('sqrt(x-x)', 'unsupported_expression'),
    ('x/(x-x+1)', 'unsupported_expression'), ('pi*x-pi*x', 'unsupported_expression'),
    ('2^3^2', 'unsupported_expression'), ('2^(3^2)', 'unsupported_expression'),
    ('x^-1', 'unsupported_expression'), ('__import__("os")', 'unsupported_expression'),
    ('nan', 'non_finite_value'), ('1/0', 'non_finite_value'),
    ('sqrt(-1)', 'non_finite_value'), ('2^9', 'math_limit_exceeded'),
    ('1e101', 'math_limit_exceeded'), ('((1e-100)^8)^8', 'math_limit_exceeded'),
    ('1e100*10/10', 'math_limit_exceeded'),
    ('x^8*x', 'math_limit_exceeded'), ('(x+1)^8', 'math_limit_exceeded'),
    ('a+b+c+d+e+f+g+h+i', 'math_limit_exceeded'),
    ('x/0', 'non_finite_value'), ('[1]', 'unsupported_expression'),
    ('x.y', 'unsupported_expression'), (r'\unknown{1}', 'unsupported_expression'),
    ('1 2', 'unsupported_expression'), ('', 'unsupported_expression'),
    ('1e-101', 'math_limit_exceeded'), ('1'*65, 'math_limit_exceeded'),
    ('-1e100*10', 'math_limit_exceeded'), ('0*(x^8*x)', 'math_limit_exceeded'),
    ('0*((1e-100)^8)^8', 'math_limit_exceeded'),
])
def test_declared_rejections(value, code):
    from evaluation.emma_math import compare_values
    result = compare_values(value, '7')
    assert result.correctness is False
    assert result.error == code


@pytest.mark.parametrize('left,right', [('pi', 'x'), ('x', 'sqrt(2)'), ('x-x', 'pi')])
def test_irrational_numeric_cannot_mix_with_symbolic_comparison(left, right):
    from evaluation.emma_math import compare_values
    assert compare_values(left, right).error == 'unsupported_expression'


@pytest.mark.parametrize('value', ['nan', '+NaN', '-INF', '+infinity', '∞', '-∞', r'\infty', r'+\infty', '$nan$', r'\(-inf\)'])
def test_nonfinite_even_on_exact_match(value):
    from evaluation.emma_math import compare_values, is_nonfinite_literal, unwrap_math
    assert is_nonfinite_literal(unwrap_math(value))
    assert compare_values(value, value).error == 'non_finite_value'


def test_wrapping_preserves_internal_content():
    from evaluation.emma_math import unwrap_math, is_nonfinite_literal, compare_values
    assert unwrap_math('  $ a b $  ') == ' a b '
    assert unwrap_math(r' \(x + x\) ') == 'x + x'
    assert unwrap_math('$$x$$') == '$x$'
    for value in ['i', 'infinite possibilities', 'NaN value', 'finite infinity']:
        assert not is_nonfinite_literal(value)
    assert compare_values('$0.5$', r'\(1/2\)').correctness
    assert not compare_values('a b', 'ab').correctness


def test_exact_long_content_bypasses_parser_limit():
    from evaluation.emma_math import compare_values
    value = 'long code choice\n' * 300
    result = compare_values(value, value)
    assert result.correctness and result.error is None


@pytest.mark.parametrize('value,right', [
    ('1' + ' '*510 + '+0', 'math_limit_exceeded'),
    ('+'*65+'1', 'math_limit_exceeded'),
    ('('*17+'1'+')'*17, 'math_limit_exceeded'),
    ('+'.join(['0']*66), 'math_limit_exceeded'),
])
def test_limits_beyond(value, right):
    from evaluation.emma_math import compare_values
    assert compare_values(value, '1.0').error == right


@pytest.mark.parametrize('value,expected', [
    ('1' + ' '*509 + '+0', '1'),
    ('('*16+'1'+')'*16, '1'),
    ('+'*16+'1', '1'),
    ('+'.join(['0']*65), '0'),
    ('2^8', '256'), ('x^8', 'x*x*x*x*x*x*x*x'),
    ('(x+1)^7', 'x^7+7*x^6+21*x^5+35*x^4+35*x^3+21*x^2+7*x+1'),
    ('a+b+c+d+e+f+g+h', 'h+g+f+e+d+c+b+a'),
    ('1'*64, '+'+'1'*64), ('1e100', '10^8*1e92'),
])
def test_limits_at_boundary(value, expected):
    from evaluation.emma_math import compare_values
    result = compare_values(value, expected)
    assert result.correctness, result


def test_oversized_power_never_reaches_constructor(monkeypatch):
    import sympy
    from evaluation.emma_math import compare_values
    original = sympy.Pow
    calls = []
    def witness(base, exponent, *args, **kwargs):
        calls.append((base, exponent))
        if getattr(base, 'is_Rational', False) and int(base.q).bit_length() > 2000:
            raise AssertionError('oversized outer power reached SymPy.Pow')
        return original(base, exponent, *args, **kwargs)
    monkeypatch.setattr(sympy, 'Pow', witness)
    assert compare_values('((1e-100)^8)^8', '0').error == 'math_limit_exceeded'
    assert any(exponent == 8 for _, exponent in calls)


def test_symbolic_coefficient_power_guard():
    from evaluation.emma_math import compare_values
    assert compare_values('((1e-100)^8*x)^8', '0').error == 'math_limit_exceeded'


def test_runtime_version_rechecked(monkeypatch):
    import sympy
    from evaluation.emma_math import compare_values
    from evaluation.schema import ContractError
    assert compare_values('1', '1').correctness
    monkeypatch.setattr(sympy, '__version__', '0.0')
    with pytest.raises(ContractError):
        compare_values('1', '1')


def test_missing_runtime_is_contract_error(monkeypatch):
    from evaluation.emma_math import require_emma_math_runtime
    from evaluation.schema import ContractError
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == 'sympy':
            raise ImportError('unavailable')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    with pytest.raises(ContractError):
        require_emma_math_runtime()


def test_import_and_scoring_are_cpu_only_and_lazy():
    script = '''
import sys
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'Reference'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from evaluation.emma_math import compare_values
assert 'sympy' not in sys.modules
assert compare_values('x+x', '2*x').correctness
'''
    result = subprocess.run([sys.executable, '-c', script], env=os.environ.copy(), text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_call_shaped_input_does_not_execute(tmp_path):
    from evaluation.emma_math import compare_values
    marker = tmp_path / 'executed'
    value = f'__import__("pathlib").Path("{marker}").touch()'
    assert compare_values(value, '0').error == 'unsupported_expression'
    assert not marker.exists()


@pytest.mark.parametrize('value', ['((1e-100*pi)^8)^8', '((1e-100*pi)^8)^2'])
def test_irrational_rational_components_checked_before_power(value, monkeypatch):
    import sympy
    from evaluation.emma_math import compare_values
    original = sympy.Pow
    def witness(base, exponent, *args, **kwargs):
        if any(int(atom.q).bit_length() > 2000 for atom in base.atoms(sympy.Rational)):
            raise AssertionError('oversized numeric component reached Pow')
        return original(base, exponent, *args, **kwargs)
    monkeypatch.setattr(sympy, 'Pow', witness)
    assert compare_values(value, '0').error == 'math_limit_exceeded'


def test_runtime_without_version_is_contract_error(monkeypatch):
    import sympy
    from evaluation.emma_math import require_emma_math_runtime
    from evaluation.schema import ContractError
    monkeypatch.delattr(sympy, '__version__')
    with pytest.raises(ContractError):
        require_emma_math_runtime()


def test_distinct_irrational_terms_do_not_combine_denominators():
    from evaluation.emma_math import compare_values
    result = compare_values('pi*(1e-100)^8+sqrt(2)*((((1/3)^8)^8)^8)^2', '0')
    assert result.correctness is True
    assert result.error is None


def test_like_irrational_terms_guard_actual_coefficient_sum(monkeypatch):
    import sympy
    from evaluation.emma_math import compare_values
    original = sympy.Add
    def witness(*args, **kwargs):
        large = [arg for arg in args if int(arg.as_coeff_Mul()[0].q).bit_length() > 1500]
        if len(large) == 2:
            raise AssertionError('oversized coefficient sum reached Add')
        return original(*args, **kwargs)
    monkeypatch.setattr(sympy, 'Add', witness)
    result = compare_values('pi*(1e-100)^8+pi*((((1/3)^8)^8)^8)^2', '0')
    assert result.error == 'math_limit_exceeded'
