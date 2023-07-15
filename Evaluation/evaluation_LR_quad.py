import numpy as np

from Utility_functions.plot_validators import *
from Models.Logistic_Regression import *
from prettytable import PrettyTable
from Models.PCA_LDA import *

def evaluation_LR_quad(DTR, LTR, DTE, LTE, l):
    for pi in [0.1, 0.5, 0.9]:
        evaluate_LR_quad(DTR, LTR, DTE, LTE, l, pi)

'''
    x = numpy.logspace(-5, 2, 20)
    y = numpy.array([])
    y_05 = numpy.array([])
    y_09 = numpy.array([])
    y_01 = numpy.array([])
    y_05_PCA = numpy.array([])
    y_09_PCA = numpy.array([])
    y_01_PCA = numpy.array([])

    for xi in x:
        scores, scoresPCA, labels = QUAD_LR_tuning(DTR, LTR, DTE, LTE, xi)
        y_09 = numpy.hstack((y_09, compute_dcf_min_effPrior(0.9, scores, labels)))
        y_05 = numpy.hstack((y_05, compute_dcf_min_effPrior(0.5, scores, labels)))
        y_01 = numpy.hstack((y_01, compute_dcf_min_effPrior(0.1, scores, labels)))
        y_09_PCA = numpy.hstack((y_09_PCA, compute_dcf_min_effPrior(0.9, scoresPCA, labels)))
        y_05_PCA = numpy.hstack((y_05_PCA, compute_dcf_min_effPrior(0.5, scoresPCA, labels)))
        y_01_PCA = numpy.hstack((y_01_PCA, compute_dcf_min_effPrior(0.1, scoresPCA, labels)))

    y = numpy.hstack((y, y_09))
    y = numpy.vstack((y, y_05))
    y = numpy.vstack((y, y_01))
    y = numpy.vstack((y, y_09_PCA))
    y = numpy.vstack((y, y_05_PCA))
    y = numpy.vstack((y, y_01_PCA))

    plot_DCF_PCA(x, y, 'lambda', 'QUAD_LR_PCA_minDCF_comparison', folder='evaluation/')
'''

def evaluate_LR_quad(DTR, LTR, DTE, LTE, l, pi):
    scores_append = []
    PCA_m9_scores = []
    PCA_m8_scores = []
    LR_labels = []

    def vecxxT(x):
       x = x[:, None]
       xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
       return xxT

    expanded_DTR = numpy.apply_along_axis(vecxxT, 0, DTR)
    expanded_DTE = numpy.apply_along_axis(vecxxT, 0, DTE)
    phi = numpy.vstack([expanded_DTR, DTR])
    phi_DTE = numpy.vstack([expanded_DTE, DTE])

    scores = quad_logistic_reg_score(phi, LTR, phi_DTE, l, pi)
    scores_append.append(scores)

    LR_labels = np.append(LR_labels, LTE, axis=0)
    LR_labels = np.hstack(LR_labels)

    # PCA m=9
    s, P = PCA(DTR, 9)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    PCA_m9_scores.append(quad_logistic_reg_score(DTR_PCA, LTR, DTE_PCA, l, pi))

    # PCA m=8
    s, P = PCA(DTR, 8)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    PCA_m8_scores.append(quad_logistic_reg_score(DTR_PCA, LTR, DTE_PCA, l, pi))

    validate_LR(scores_append, LR_labels, 'LR QUAD, RAW data', l, pi)

    validate_LR(PCA_m9_scores, LR_labels, 'LR QUAD, PCA m=9', l, pi)

    validate_LR(PCA_m8_scores, LR_labels, 'LR_QUAD, PCA_m8', l, pi)

def validate_LR(scores, LR_labels, appendToTitle, l, pi):
    scores_append = np.hstack(scores)
    scores_tot_05 = compute_dcf_min_effPrior(0.1, scores_append, LR_labels)
    scores_tot_01 = compute_dcf_min_effPrior(0.5, scores_append, LR_labels)
    scores_tot_09 = compute_dcf_min_effPrior(0.9, scores_append, LR_labels)

    t = PrettyTable(["Type", "π=0.1", "π=0.5", "π=0.9"])
    t.title = appendToTitle
    t.add_row(['QUAD_LR, lambda=' + str(l) + " π_t=" + str(pi), round(scores_tot_05, 3), round(scores_tot_01, 3), round(scores_tot_09, 3)])
    print(t)


def QUAD_LR_tuning(DTR, LTR, DTE, LTE, l):
    scores_append = []
    PCA_m9_scores = []
    LR_labels = []

    def vecxxT(x):
        x = x[:, None]
        xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
        return xxT

    expanded_DTR = numpy.apply_along_axis(vecxxT, 0, DTR)
    expanded_DTE = numpy.apply_along_axis(vecxxT, 0, DTE)
    phi = numpy.vstack([expanded_DTR, DTR])
    phi_DTE = numpy.vstack([expanded_DTE, DTE])

    scores_append.append(quad_logistic_reg_score(phi, LTR, phi_DTE, l, 0.5))

    # PCA m=9
    s, P = PCA(DTR, 9)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    PCA_m9_scores.append(quad_logistic_reg_score(DTR_PCA, LTR, DTE_PCA, l, 0.5))

    LR_labels = np.append(LR_labels, LTE, axis=0)
    LR_labels = np.hstack(LR_labels)

    return np.hstack(scores_append), np.hstack(PCA_m9_scores), LR_labels