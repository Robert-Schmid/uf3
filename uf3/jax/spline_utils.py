from math import degrees
import ndsplines
import numpy as np


def smooth_spline_to_zero(spline: ndsplines.NDSpline):
    """
    Adds additional basis splines with zero coefficients to force standart boundary conditions.
    Where the spline goes smoothly to zero at the borders of the knot range.
    """
    if not spline.ydim == 1:
        raise NotImplementedError("Higher y dimensions are not conisdered.")

    ks = spline.knots
    c = spline.coefficients
    deg = spline.degrees
    xdim = spline.xdim

    knots = []
    padding = []
    for i in range(xdim):
        knots.append(ks[i], (deg[i], deg[i]), "edge")
        padding.append((deg[i], deg[i]))

    coeff = np.pad(c, padding)

    return ndsplines.NDSpline(knots, coeff, deg, spline.periodic, spline.extrapolate)
