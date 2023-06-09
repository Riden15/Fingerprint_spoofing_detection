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


def Poly_KernelFunction(DTR, LTR, DTE, C, constant, K, degree):
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    aStar, loss = train_SVM_polynomial(DTR, LTR, C, constant, K, degree)
    kernel = (numpy.dot(DTR.T, DTE) + constant) ** degree + K * K
    score = numpy.sum(numpy.dot(aStar * vrow(Z), kernel), axis=0)
    return score


def train_SVM_polynomial(DTR, LTR, C, constant, K, degree):
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = (numpy.dot(DTR.T, DTR) + constant) ** degree + K ** 2
    H = mcol(Z) * vrow(Z) * H

    alphaStar, JDual, LDual = calculate_lbgf(H, DTR, C)

    return alphaStar, JDual(alphaStar)[0]


def RBF_KernelFunction(DTR, LTR, DTE, C, K, gamma):
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    aStar, loss = train_SVM_RBF(DTR, LTR, C, K, gamma)

    kern = numpy.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            kern[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(DTR[:, i] - DTE[:, j]) ** 2)) + K * K

    score = numpy.sum(numpy.dot(aStar * vrow(Z), kern), axis=0)
    return score


def train_SVM_RBF(DTR, LTR, C, K=1, gamma=1.):
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    # kernel function
    kernel = numpy.zeros((DTR.shape[1], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            kernel[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(DTR[:, i] - DTR[:, j]) ** 2)) + K * K
    H = mcol(Z) * vrow(Z) * kernel
    alphaStar, JDual, LDual = calculate_lbgf(H, DTR, C)

    return alphaStar, JDual(alphaStar)[0]
