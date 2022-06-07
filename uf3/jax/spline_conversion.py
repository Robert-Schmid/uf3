import numpy as np
import jax.numpy as jnp
import math
from scipy.interpolate import BSpline



def spline_value_recurrence(delta, K, x, coefficients, knots):
    """
    From the book "An introduction to the Use of Splines in Computer Graphics"
    by Richard H. Bartels, John C. Beatty
    Page 206(print)/211(pdf)
    """

    max_i = len(coefficients) + K
    max_j = K
    C = np.zeros((max_j, max_i), dtype=np.float64)

    C[0, 0 : len(coefficients)] = coefficients

    for j in range(1, K):
        for i in range(max(0, delta - K + j + 1), delta + 1):

            C_1, C_2 = 0.0, 0.0
            if j > 0 and j <= max_j:
                if i >= 0 and i < max_i:
                    C_1 = C[j - 1, i]
                if i > 0 and i <= max_i:
                    C_2 = C[j - 1, i - 1]

            C[j, i] = ((x - knots[i]) / (knots[i + K - j] - knots[i])) * C_1

            C[j, i] = (
                C[j, i] + ((knots[i + K - j] - x) / (knots[i + K - j] - knots[i])) * C_2
            )

    return C[K - 1, delta]


def segment_polynomial_conversion_recurrence(delta, k, t, c):
    """
    From the book "An introduction to the Use of Splines in Computer Graphics"
    by Richard H. Bartels, John C. Beatty
    Page 207(print)/212(pdf)

    but more high-level. This introduces duplicate work in spline and derivative computations,
    but easier to understand and maintain and sufficiently fast.
    """

    #TODO deal with the borders by artificially extending the knots and add splines with 0 coefficients
    #TODO deal with duplicate knots
    a = np.zeros(k + 1)

    s = BSpline(t, c, k, False)

    a[0] = s(t[delta])

    for r in range(1, k + 1):
        sp = s.derivative(r)
        a[r] = sp(t[delta])
        a[r] = a[r] / math.factorial(r)

    return a


def spline_polynomial_coeffients(t, c, k):
    a = np.zeros((len(t) - 2 * k, k + 1), order="F")

    for i in range(k, len(t) - k):
        a[i - k, :] = segment_polynomial_conversion_recurrence(i, k, t, c)

    return a


def evaluate_segment_polynomials(a, t, x, segment_map=None):
    # TODO make sure a, x and t are numpy arrays
    k = 3
    f = equidistant_map
    if segment_map != None:
        # TODO provide fast maps for arbitrary knot spacing
        f = segment_map
    idx, x = f(x, t)
    idx = idx - k

    _, n = a.shape

    x0 = a[idx, -1] + x * 0
    for i in range(2, n + 1):
        x0 = a[idx, -i] + x0 * x
    return x0


def equidistant_map(x, t):
    delta = t[1] - t[0]
    idx = (x - t[0]) // delta
    idx = idx.astype(np.int32)
    x = x - t[idx]

    return (idx, x)

#TODO add a map with control flow for simplicity

#TODO add a naive iterative map - probably really fast for few knots


def evaluate_segment_polynomials_jax(a, t, x, segment_map=None):
    # TODO make sure a, x and t are jax arrays
    a = jnp.asarray(a)
    k = 3
    f = equidistant_map_jax
    if segment_map != None:
        # TODO provide fast maps for arbitrary knot spacing
        f = segment_map
    idx, x = f(x, t)
    idx = idx - k

    _, n = a.shape

    x0 = a[idx, -1] + x * 0
    for i in range(2, n + 1):
        x0 = a[idx, -i] + x0 * x
    return x0


def equidistant_map_jax(x, t):
    delta = t[1] - t[0]
    idx = (x - t[0]) // delta
    idx = idx.astype(jnp.int32)
    x = x - t[idx]

    return (idx, x)


def main():
    t = [1, 2, 3, 4, 5, 6, 7, 8]
    c = [1, 2, 3, 4]
    # co=[1,1,1,1]
    k = 4

    delta = 3
    x = 4.5

    spl = BSpline(t, c, k - 1)
    res = spline_value_recurrence(delta, k, x, c, t)

    a = spline_polynomial_coeffients(t, c, 3)
    print(a)
    res2 = evaluate_segment_polynomials(a, np.asarray(t), np.asarray([x, 5.5]))

    print(f"{res2} compared to {spl([x,5.5])}")

    # poly_co = segment_polynomial_conversion_recurrence(3, k - 1, co, knots)

    # print(poly_co)


if __name__ == "__main__":
    main()
