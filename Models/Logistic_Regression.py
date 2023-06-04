import numpy
import scipy

def logreg_obj(v, DTR, LTR, l):
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

    logreg_obj(param, DTR, LTR, l)
    x, d, f = scipy.optimize.fmin_l_bfgs_b(logreg_obj, param, approx_grad=True, args=(DTR, LTR, l))

    w = x[0:-1]  # tutti tranne l'ultimo
    b = x[-1]  # ultimo valore di x
    S = numpy.dot(w.T, DTE) + b
    return S