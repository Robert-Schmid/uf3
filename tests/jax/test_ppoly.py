import pytest

from uf3.jax.ppoly import *
import jax.numpy as jnp
import numpy as np
import ndsplines
from utils import make_random_spline
from numpy.testing import assert_allclose

def test_init():
    c = jnp.asarray([[2., 3.], [1., 1.], [0., 0.],[0., 0.]])
    x = [jnp.asarray([4,5,6])]

    pp = PPoly(c, x, deg=[3])

def test_1d():
    spline = make_random_spline(1, 3) # generate 1 D spline of degree 3

    ppoly = PPoly.from_NDSpline(spline)

    x = np.random.rand(10)

    assert_allclose(spline(x), ppoly(x))

def test_2d():
    spline = make_random_spline(2, 3) # generate 2 D spline of degree 3 in both dimensions

    ppoly = PPoly.from_NDSpline(spline)

    x = np.random.rand(10)

    assert_allclose(spline(x), ppoly(x))

def test_3d():
    spline = make_random_spline(3,3)

    ppoly = PPoly.from_NDSpline(spline)





# for interpreter
# 3D

# t = np.asarray([[1,0.5,4],[2,1,6],[3,1.5,8],[4,2,0],[5,2.5,0],[5,3,8]])

# xp = np.asarray([[3.3,1.4,7],[2.5,0.7,6.1],[4.9,1,4.1]])

# idx = (xp - t[0]) // (t[1] - t[0])
# idx = idx.astype(np.int32)
# n, d = xp.shape

# cxp = xp - t[idx, np.tile(np.arange(d, dtype=np.int32), (n, 1))]

# id = tuple(np.transpose(idx))

# c = [[[[1,2,3,4],[2,3,4,5]],
#           [[3,4,5,6],[4,5,6,7]]],
#          [[[3,2,3,4],[4,3,4,5]],
#           [[5,4,5,6],[6,5,6,7]]],
#           [[[1,4,3,4],[2,5,4,5]],
#           [[3,6,5,6],[4,7,6,7]]],
#          [[[1,2,3,6],[2,3,4,7]],
#           [[3,4,5,8],[4,5,6,9]]]]
# c = np.asarray(c)

# # 1D
# t = np.asarray([[0.5],[1],[1.5],[2],[2.5],[3]])

# xp = np.asarray([[1.6],[0.7],[2.9]])

# idx = (xp - t[0]) // (t[1] - t[0])
# idx = idx.astype(np.int32)
# n, d = xp.shape

# cxp = xp - t[idx, np.tile(np.arange(d, dtype=np.int32), (n, 1))]


############################

# vars for testing in the interpreter
# x = [[1,2], [2,3], [3,4]]
# x = np.asarray(x)

# c = [[[1,1,1,1,1,1,1,1,1,1,1,1],[1,2,1,2,1,2,1,2,1,2,1,2]],
# [[2,2,2,2,2,2,2,2,2,2,2,2],[3,3,3,3,3,3,3,3,3,3,3,3]]
# ]
# c = np.asarray(c)

# e = fun_test.coefficient_enumeration((3,2))

# idx = [[0,0],[0,1],[1,1]]
# idx = np.asarray(idx, dtype=np.int32)

############################
# for ndsplines

# import ndsplines
# import numpy as np

# xknots = [0,1,2,3,4,5,6,7,8]
# yknots = [3,4,5,6,7,8,9,10,11]

# xknots = np.asarray(xknots)
# yknots = np.asarray(yknots)

# knots = [xknots, yknots]

# degrees = np.asarray([3,2])

# coeff = [[1,1,1,1,1,1],[1,2,1,1,1,1],[1,1,2,1,1,1],[1,1,1,2,1,1],[1,1,1,1,2,1]]
# coeff = np.asarray(coeff)