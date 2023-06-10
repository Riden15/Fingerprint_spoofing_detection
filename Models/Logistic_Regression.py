import numpy
import scipy
from Utility_functions.General_functions import *

def linear_logreg_obj(v, DTR, LTR, l):
    w = v[0:-1]
    b = v[-1]
    first = l / 2 * numpy.power(numpy.linalg.norm(w), 2)
    second = 1 / DTR.shape[1]
    add = 0
    for i in range(DTR.shape[1]):
        z = 2 * LTR[i] - 1
        add += numpy.logaddexp(0, -z * (numpy.dot(w.T, DTR[:, i]) + b))
    return first + second * add

def lr_binary(DTR, LTR, DTE, l):
    param = numpy.zeros(DTR.shape[0] + 1)

    linear_logreg_obj(param, DTR, LTR, l)
    x, d, f = scipy.optimize.fmin_l_bfgs_b(linear_logreg_obj, param, approx_grad=True, args=(DTR, LTR, l))

    w = x[0:-1]  # tutti tranne l'ultimo
    b = x[-1]  # ultimo valore di x
    S = numpy.dot(w.T, DTE) + b
    return S

def quad_logreg_obj(DTR, LTR, l, pi):
    M = DTR.shape[0]
    Z = LTR * 2.0 - 1.0

    def logreg_obj(v):
        w = mcol(v[0:M])
        b = v[-1]
        reg = 0.5 * l * numpy.linalg.norm(w) ** 2
        s = (numpy.dot(w.T, DTR) + b).ravel()
        nt = DTR[:, LTR == 0].shape[1]
        avg_risk_0 = (numpy.logaddexp(0, -s[LTR == 0] * Z[LTR == 0])).sum()
        avg_risk_1 = (numpy.logaddexp(0, -s[LTR == 1] * Z[LTR == 1])).sum()
        return reg + (pi / nt) * avg_risk_1 + (1 - pi) / (DTR.shape[1] - nt) * avg_risk_0

    return logreg_obj

def quad_logistic_reg_score(DTR, LTR, DTE, l, pi=0.5):
    logreg_obj = quad_logreg_obj(numpy.array(DTR), LTR, l, pi)
    x, d, f = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0] + 1), approx_grad=True)
    w = x[0:DTR.shape[0]]
    b = x[-1]
    STE = numpy.dot(w.T, DTE) + b
    return STE