import pytest

from uf3.jax.spline_conversion import *

"""
Things to test:
- equidistant knots
- lower boundary
- upper boundary
- duplicate knots
- out of bounds
- extrapolation
- acuraccy?
- 3-D
"""


def test_equal_knot_spacing():
    t = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
    c = np.asarray([1, 2, 3, 4])
    deg = 3

    x = np.asarray([4.4, 5.5])

    spl = BSpline(t, c, deg)
    poly = spline_polynomial_coeffients(t, c, deg)

    assert np.allclose(spl(x), evaluate_segment_polynomials(poly, t, x))


def test_lower_boundary():
    t = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
    c = np.asarray([1, 2, 3, 4])
    deg = 3

    x = np.asarray([1.0, 1.4, 2.5, 3.6])

    spl = BSpline(t, c, deg)
    poly = spline_polynomial_coeffients(t, c, deg)

    assert np.allclose(spl(x), evaluate_segment_polynomials(poly, t, x))


def test_quadruple_end_knots():
    t = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
    c = np.asarray([1, 2, 3, 4])
    deg = 3

    x = np.asarray([5.6, 6.3, 7.999])

    spl = BSpline(t, c, deg)
    poly = spline_polynomial_coeffients(t, c, deg)

    assert np.allclose(spl(x), evaluate_segment_polynomials(poly, t, x))


def test_quadruple_knots():
    t = np.asarray([1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6])
    c = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
    deg = 3

    x = np.asarray([2.1, 4.4, 5.5])

    spl = BSpline(t, c, deg)
    poly = spline_polynomial_coeffients(t, c, deg)

    assert np.allclose(spl(x), evaluate_segment_polynomials(poly, t, x))


def test_out_of_bounds():
    t = np.asarray([1, 2, 3, 4, 5, 6, 7, 8])
    c = np.asarray([1, 2, 3, 4])
    deg = 3

    x = np.asarray([0.5, 8.0, 8.7])

    sol = np.zeros_like(x)
    poly = spline_polynomial_coeffients(t, c, deg)

    assert np.testing.assert_equal(sol, evaluate_segment_polynomials(poly, t, x))
