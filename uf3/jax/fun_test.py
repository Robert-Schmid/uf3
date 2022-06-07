import math
import numpy as np
import ndsplines

a = 4

def coefficient_enumeration(order):
    d = len(order)
    
    if d == 1:
        poly = [[0]] * order[0]
        poly = poly + [[]]
        return [poly]

    if d == 2:
        result = []
        for i in range(0, order[1]+1):
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


"""
x : inputs in the local power basis x = x' - t_i where t_i <= x' < t_i+1 and original input x'
coeff : polynomial coefficients for each breakpoint interval
enum : coefficient enumeration
idx: breakpoint intervals for each x used to choose coefficients
"""
def compute_enum(x, coeff, enum, idx):
    
    id = tuple(np.transpose(idx))
    c = coeff[id]

    result = np.zeros(len(x))
    i = 0
    for p in enum:
        tmp = np.ones_like(result)
        for q in p:
            tmp *= c[:,i] * np.prod(x[:,q], axis=1)
            i += 1
        result += tmp
    
    return result


def enumeration_to_derivative_order(enum, degree):
    order = []
    for e in enum:
        total=[0] * degree
        for i in e:
            for j in i:
                total[j] += 1
        
        for i in e:
            order.append(total.copy())
            for j in i:
                total[j] -= 1

    return order


def spline_segment(spline: ndsplines.NDSpline, support, coeff_order):
    c = np.zeros(len(coeff_order))

    for i, order in enumerate(coeff_order):
        # get the correct derivative for the coefficient
        s = spline
        for j, o in enumerate(order):
            s = s.derivative(j, o)
        
        divisor = 1
        for o in order:
            divisor *= math.factorial(o)
        
        c[i] = s(support) / divisor

    return c


def single_poly_eval(coefficients, x, support, coeff_order):
    x = x - support

    result = np.zeros(1)
    i = 0
    for p in coeff_order:
        tmp = np.zeros_like(result)
        for q in p:
            #tmp += coefficients[i] 
            #tmp *= np.prod(x[q])
            tmp = (tmp + coefficients[i]) * np.prod(x[q])
            i += 1
        result += tmp
    
    return result


def test_2d():
    a = coefficient_enumeration((3,2))
    print(a)
    b = enumeration_to_derivative_order(a,2)
    print(b)

    # create 2-D spline
    xknots = [0,1,2,3,4,5,6,7,8]
    yknots = [3,4,5,6,7,8,9,10,11]

    xknots = np.asarray(xknots)
    yknots = np.asarray(yknots)

    knots = [xknots, yknots]
    degrees = np.asarray([3,2])

    coefficients = [[1,1,1,1,1,1],[1,2,1,1,1,1],[1,1,2,1,1,1],[1,1,1,2,1,1],[1,1,1,1,2,1]]
    coefficients = np.asarray(coefficients)

    s = ndsplines.NDSpline(knots, coefficients, degrees, extrapolate=False)

    support = np.asarray([3.0,5.0])

    c = spline_segment(s, support, b)

    print(c)

    x = np.asarray([3.2, 5.4])

    test = single_poly_eval(c, x, support, a)
    print(test)
    correct = s(x)
    print(correct)

def test_1d():
    a = coefficient_enumeration((3,))
    print(a)
    b = enumeration_to_derivative_order(a,1)
    print(b)

    # create 1-D spline
    xknots = [0,1,2,3,4,5,6,7,8]

    xknots = np.asarray(xknots)

    knots = [xknots]
    degrees = np.asarray([3])

    coefficients = [1,2,3,4,5]
    coefficients = np.asarray(coefficients)

    s = ndsplines.NDSpline(knots, coefficients, degrees, extrapolate=False)

    support = np.asarray([3.0])

    c = spline_segment(s, support, b)

    print(c)

    x = np.asarray([3.2])

    test = single_poly_eval(c, x, support, a)
    print(test)
    correct = s(x)
    print(correct)

if __name__ == "__main__":
    test_2d()