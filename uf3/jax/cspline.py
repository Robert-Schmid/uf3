from jax import lax
import jax.numpy as np

def cubic_spline_intervals(knots,
                           interval):
    '''
    knots: [t0,t1,t2,t3,t4]
    interval: 0 for [t0,t1]
              1 for [t1,t2]
              2 for [t2,t3]
              3 for [t3,t4]
    '''
    t = knots
    if t[interval] == t[interval + 1]:
        #raise ArithmeticError("No function between two equal knots.")
        return lambda x: x * 0
    
    if interval == 0:
        tmp = (t[3]-t[0]) * (t[2]-t[0]) * (t[1]-t[0])
        return (lambda x: pow(x-t[0],3) / tmp )
    if interval == 1:
        return (lambda x: (x-t[0]) / (t[3]-t[0]) * ( (x-t[0]) / (t[2]-t[0]) * (t[2]-x) / (t[2]-t[1]) + (t[3]-x) / (t[3]-t[1]) * (x-t[1]) / (t[2]-t[1]) ) + (t[4]-x) / (t[4]-t[1]) * (x-t[1]) / (t[3]-t[1]) * (x-t[1]) / (t[2]-t[1]))
    if interval == 2:
        return (lambda x: (x-t[0]) / (t[3]-t[0]) * (t[3]-x) / (t[3]-t[1]) * (t[3]-x) / (t[3]-t[2]) + (t[4]-x) / (t[4]-t[1]) * ( (x-t[1]) / (t[3]-t[1]) * (t[3]-x) / (t[3]-t[2]) + (t[4]-x) / (t[4]-t[2]) * (x-t[2]) / (t[3]-t[2]) ) )
    if interval == 3:
        tmp = (t[4]-t[1]) * (t[4]-t[2]) * (t[4]-t[3])
        return (lambda x: pow(t[4]-x,3) / tmp )

def cubic_spline_segments(knots,
                 coefficients,
                 index):
    '''
    knots: [t0,t1,t2,t3,t4,t5,t6,t7,...]
    coefficients: [c0,c1,c2,c3,...]
    
    gives a function for the cubic spline in interval [t_index,t_index+1]
    '''
    if len(knots) != len(coefficients) + 4:
        raise IndexError("There must be n+4 knots for n coefficients")
    if index >= len(knots)-1 or index < 0:
        raise IndexError("Index must be 0 <= index < len(knots) - 1")
    
    c = coefficients
        
    if index >= 0 and len(knots) >= index + 5:
        tmp0 = cubic_spline_intervals(knots[index:index+5],0)
        bs_0 = lambda x: c[index] * tmp0(x)
    else:
        bs_0 = lambda x: 0 * x
    if index >= 1 and len(knots) >= index + 4:
        tmp1 = cubic_spline_intervals(knots[index-1:index+4],1)
        bs_1 = lambda x: c[index-1] * tmp1(x)
    else:
        bs_1 = lambda x: 0 * x
    if index >= 2 and len(knots) >= index + 3:
        tmp2 = cubic_spline_intervals(knots[index-2:index+3],2)
        bs_2 = lambda x: c[index-2] * tmp2(x)
    else:
        bs_2 = lambda x: 0 * x
    if index >= 3 and len(knots) >= index + 2:
        tmp3 = cubic_spline_intervals(knots[index-3:index+2],3)
        bs_3 = lambda x: c[index-3] * tmp3(x)
    else:
        bs_3 = lambda x: 0 * x
    
    combined = (lambda x: bs_3(x) + bs_2(x) + bs_1(x) + bs_0(x) )
    
    return (combined, [bs_0,bs_1,bs_2,bs_3])


def cubic_spline(knots, coefficients):

    segments = []
    segment_knots = []
    
    for i, k in enumerate(knots[:-1]):
        # double knot => empty interval
        if knots[i+1] == k:
            continue

        fun, _ = cubic_spline_segments(knots, coefficients, i)
        segments.append(fun)
        segment_knots.append(k)

    segment_knots.append(knots[-1])
    # build a complete function with lax conditionals
    seg_knots = np.asarray(segment_knots)

    def spline(x):
        result = []
        
        for i in range(1, len(seg_knots)):
            result.append(np.where(x < seg_knots[i],
                                 np.where(x >= seg_knots[i-1],
                                          segments[i-1](x),
                                          0.),
                                 0.))
        return np.asarray(result).sum()
    return spline