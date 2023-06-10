import numpy
import scipy
import sklearn
from Utility_functions.General_functions import *

def calculate_lbgf(H, DTR, C):
    def JDual(alpha):
        Ha = numpy.dot(H, mcol(alpha))
        aHa = numpy.dot(vrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        numpy.zeros(DTR.shape[1]),
        bounds=[(0, C)] * DTR.shape[1],
        factr=1.0,
        maxiter=10000,
        maxfun=100000,
    )

    return alphaStar, JDual, LDual

def train_SVM_linear(DTR, LTR, K, C):
    DTREXT = numpy.vstack([DTR, K * numpy.ones((1, DTR.shape[1]))])
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = numpy.dot(DTREXT.T, DTREXT)
    H = mcol(Z) * vrow(Z) * H

    def JPrimal(w):
        S = numpy.dot(vrow(w), DTREXT)
        loss = numpy.maximum(numpy.zeros(S.shape), 1 - Z * S).sum()
        return 0.5 * numpy.linalg.norm(w) ** 2 + C * loss

    alphaStar, JDual, LDual = calculate_lbgf(H, DTR, C)
    wStar = numpy.dot(DTREXT, mcol(alphaStar) * mcol(Z))
    return wStar, JPrimal(wStar)

def single_F_POLY(DTR, LTR, DTE, LTE, L, C, K, costant=1.0, degree=2):

    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    aStar, loss = train_SVM_polynomial(DTR, LTR, C=1.0, constant=costant, degree=degree, K=K)
    kernel = (numpy.dot(DTR.T, DTE) + costant) ** degree + K * K
    score = numpy.sum(numpy.dot(aStar * vrow(Z), kernel), axis=0)

    errorRate = (1 - numpy.sum((score > 0) == LTE) / len(LTE)) * 100
    print("K = %d, costant = %d, loss = %e, error =  %.1f" % (K, costant, loss, errorRate))


def train_SVM_polynomial(DTR, LTR, C, K, constant, degree):
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = (numpy.dot(DTR.T, DTR) + constant) ** degree + K ** 2
    # Dist = mcol((DTR**2).sum(0)) + mrow((DTR**2).sum(0)) - 2*numpy.dot(DTR.T, DTR)
    # H = numpy.exp(-Dist)
    H = mcol(Z) * vrow(Z) * H

    alphaStar, JDual, LDual = calculate_lbgf(H, DTR, C)

    return alphaStar, JDual(alphaStar)[0]