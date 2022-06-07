from argparse import ArgumentError

from jax import jit
import jax.numpy as jnp
import numpy as np
import ndsplines
from typing import Tuple
import math
import itertools


class PPoly:
    """
    N-dimensional piecewise polynomial in terms of coefficients and breakpoints

    The polynomial between x[n][i] and x[n][i+1] is written in the local power basis:

        S = sum( c[m1, ..., mN, i, j, ...] * (xp - x[])) #TODO more documentation

    
    Parameters
    ----------
    c : jax ndarray, shape (k1 * k2 * ..., m1, m2, ...)
        Polynomial coefficients, order `ki` in dimension i and `mi` intervals in dimension i.
    x : list of jax arrays,
        Polynomial breakpoints for each dimension. Must be sorted in increasing order.
    extrapolate: currently unsuported
    deg: list of polynomial degrees.
        Used for additional sanity checks
    equidistant_intervals: if distances between breakpoints are constant.
        Currently only True is supported, will be removed and automatically detected
        in later versions
    """

    def __init__(self, c, x, extrapolate=False, deg=None, equidistant_intervals=True, naive=False):
        self.x = x
        self.xdim = len(x)
        self.equidistant = equidistant_intervals
        x_max = 0

        for i in self.x:
            if not len(i.shape) == 1:
                raise ValueError("Breakpoint arrays must be one-dimensional.")
            if x_max < len(i):
                x_max = len(i)

        self.x_max = x_max

        self.c = c
        if not self.c.ndim == 1 + self.xdim:
            raise ValueError(
                "Coefficient array must have one additional dimension for each breakpoint dimension."
            )

        if not deg == None:
            deg = jnp.asarray(deg)
            if not len(deg) == self.xdim:
                raise ValueError("deg list has to specify a degree for each dimension.")
            if not np.product(deg + 1) == self.c.shape[0]:
                raise ValueError(
                    "Coefficients inconsistent with individual polynomial degrees."
                )
        else:
            # TODO compute degrees form coefficient array
            pass

        self.deg = deg
        if not naive:
            self.enum = coefficient_enumeration(
                self.deg
            )  # TODO make sure deg is formed correctly
        else:
            self.enum = naive_coefficient_enumeration(self.deg)

        # store breakpoints in a single 2d array for more efficient access
        # dimensions are xdim * length of the longest breakpoint array
        # shorter breakpoint dimensions are filled with 0
        # the last entry is always the biggest breakpoint instead of 0

        t = np.zeros((self.x_max, self.xdim))
        for i, xi in enumerate(self.x):
            t[: len(xi), i] = xi
            t[-1, i] = xi[-1]

        self.t = jnp.asarray(t)

    def __call__(self, x):
        # TODO handle one n-d input vs n 1-d inputs here and reshape appropriately
        x = jnp.asarray(x)
        reshaped = False
        if len(x.shape) == 0:  # just a number as input
            x = jnp.reshape(x, (-1, 1))
            reshaped = True
        elif (
            len(x.shape) == 1
        ):  # array of numbers might be one n-d input or n 1-d inputs
            reshaped = True
            if self.xdim == 1:
                x = jnp.reshape(x, (-1, 1))
            else:
                if not len(x) == self.xdim:
                    raise ArgumentError("Wrong input dimensionality.")
                x = jnp.reshape(x, (-1, self.xdim))

        if self.equidistant:
            f = self._evaluate_equidistant()
            result = f(jnp.asarray(x))
            if reshaped:
                return result.flatten()
            else:
                return result
        else:
            raise NotImplementedError("Currently only equidistant breakpoints.")

    def _evaluate(self):
        def evaluation(xp: jnp.ndarray):
            pass

        return evaluation

    def _evaluate_equidistant(self):
        # assumes properly shaped inputs
        base = self.t[0]
        deltas = self.t[1] - self.t[0]

        # @jit
        def equidistant_evaluation(xp: jnp.ndarray):
            # 1st step:
            #   find breakpoints for xp and compute xp - x
            idx = (xp - base) // deltas
            idx = idx.astype(jnp.int32)
            n, d = xp.shape
            # selecting correct rows from t with indexing and the help of a column index array
            # TODO check, that the column array doesn't actually get created and stored
            # more info: https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
            x = xp - self.t[idx, jnp.tile(jnp.arange(d, dtype=jnp.int32), (n, 1))]

            # 2nd step:
            #   multiply xp with the correct coefficients to compute the polynomials

            id = (slice(None),) + tuple(jnp.transpose(idx))
            c = self.c[id]

            result = jnp.zeros(len(x))
            i = 0
            for p in self.enum:
                tmp = jnp.zeros_like(result)
                for q in p:
                    tmp = (tmp + c[i, :]) * jnp.prod(x[:, q], axis=1)
                    i += 1
                result += tmp

            return result

        return equidistant_evaluation

    def from_NDSpline(spline: ndsplines.NDSpline, naive=False):
        # only 1&2 D with efficient coefficient ordering

        if spline.ydim > 1:
            raise ArgumentError("Only splines with scalar output are supported.")

        deg = jnp.asarray(spline.degrees)
        xdim = len(deg)
        coefficient_shape = [np.prod(spline.degrees + 1)]

        distance_test = True
        knots = []
        for k in spline.knots:
            # remove duplicate knots
            kn = np.unique(k)
            # test if knots are equidistant
            if distance_test:
                step = kn[1] - kn[0]
                for i in range(1, len(kn)):
                    distance_test = distance_test and (
                        math.isclose(kn[i] - kn[i - 1], step)
                    )  # does need some epsilon allowance
                    if not distance_test:
                        break

            knots.append(jnp.asarray(kn))
            coefficient_shape.append(len(kn) - 1)

        c = np.zeros(tuple(coefficient_shape))

        # order the coefficients correctly
        if not naive:
            enum = coefficient_enumeration(tuple(deg))
        else:
            enum = naive_coefficient_enumeration(tuple(deg))

        coeff_order = enumeration_to_derivative_order(enum, len(deg))

        # for all polynomial pieces
        for piece in np.ndindex(c.shape[1:]):
            # get a 1-D view of the relevant polynomial coefficients
            idx = (slice(None),) + piece
            coeff = c[idx]

            # calculate the base knots for this section
            support = [0] * xdim
            for i in range(xdim):
                support[i] = knots[i][piece[i]]

            # calculate polynomial coefficients from spline

            for i, order in enumerate(coeff_order):
                # get the correct derivative for the coefficient
                s = spline
                for j, o in enumerate(order):
                    s = s.derivative(j, o)

                divisor = 1
                for o in order:
                    divisor *= math.factorial(o)

                coeff[i] = s(support) / divisor

        return PPoly(
            jnp.asarray(c), knots, deg=deg, equidistant_intervals=distance_test, naive=naive
        )


# for test input (a,b,c,...) -> len(flatten(result)) == a*b*c*...
def coefficient_enumeration(order):
    d = len(order)

    if d == 1:
        poly = [[0]] * order[0]
        poly = poly + [[]]
        return [poly]

    if d == 2:
        result = []
        for i in range(0, order[1] + 1):
            tmp = []
            state = np.asarray((order[0] - i, order[1]))

            for m in range(order[1], i, -1):
                tmp.append([1])

            for n in range(order[0] - i, 0, -1):
                tmp.append([0])

            state = state - [order[0] - i, order[1] - i]
            remainingA = [0] * state[0]
            remainingB = [1] * state[1]
            tmp.append(remainingA + remainingB)
            result.append(tmp)

        return result

    raise NotImplementedError("Only up to 2D supported yet")


# for test input (a,b,c,...) -> len(result) == a*b*c*...
def naive_coefficient_enumeration(order):
    d = len(order)

    polys = ()

    for x, o in enumerate(order):
        tmp = []
        for i in range(
            o, -1, -1
        ):  # start with higher potencys for readability, i.e. a^3 a^2 ...
            tmp = tmp + [[x] * i]
        polys = polys + (tmp,)

    enum = list(itertools.product(*polys))
    enum = [[list(itertools.chain(*e))] for e in enum]

    return enum


def enumeration_to_derivative_order(enum, ndim):
    order = []
    for e in enum:
        total = [0] * ndim
        for i in e:
            for j in i:
                total[j] += 1

        for i in e:
            order.append(total.copy())
            for j in i:
                total[j] -= 1

    return order
