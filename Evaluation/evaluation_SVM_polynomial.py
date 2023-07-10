import sys
import numpy as np
from prettytable import PrettyTable

from Utility_functions.plot_validators import plot_DCF_PCA

sys.path.append('../')
from Utility_functions.Validators import *
from Models.SVM import *
from Models.PCA_LDA import *


def evaluation_SVM_polynomial(DTR, LTR, DTE, LTE, K, C, constant, degree):
    evaluate_SVM_polynomial(DTR, LTR, DTE, LTE, K, C, constant, degree)

'''
    x = numpy.logspace(-5, 1, 15)
    y = numpy.array([])
    y_05 = numpy.array([])
    y_09 = numpy.array([])
    y_01 = numpy.array([])
    y_05_PCA = numpy.array([])
    y_09_PCA = numpy.array([])
    y_01_PCA = numpy.array([])
    for xi in x:                                                            # C   c  K
        scores, scoresPCA, labels = SVM_polynomial_tuning(DTR, LTR, DTE, LTE, xi, 1, 1.0, 2)
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
    plot_DCF_PCA(x, y, 'C', 'SVM_Poly_minDCF_comparison_K=1_c=1_d=2', folder='evaluation/')

    y = numpy.array([])
    y_05 = numpy.array([])
    y_09 = numpy.array([])
    y_01 = numpy.array([])
    y_05_PCA = numpy.array([])
    y_09_PCA = numpy.array([])
    y_01_PCA = numpy.array([])
    for xi in x:                                                             # C   c  K
        scores, scoresPCA, labels = SVM_polynomial_tuning(DTR, LTR, DTE, LTE, xi, 1, 1.0, 3)
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
    plot_DCF_PCA(x, y, 'C', 'SVM_Poly_minDCF_comparison_K=1_c=1_d=3', folder='evaluation/')
'''

def evaluate_SVM_polynomial(DTR, LTR, DTE, LTE, K, C, constant, degree):
    scores_append = []
    PCA_m9_scores = []
    PCA_m8_scores = []
    SVM_labels = []

    score = Poly_KernelFunction(DTR, LTR, DTE, C, constant, K, degree)
    scores_append.append(score)

    # PCA m=9
    s, P = PCA(DTR, m=9)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    score = Poly_KernelFunction(DTR_PCA, LTR, DTE_PCA, C, constant, K, degree)
    PCA_m9_scores.append(score)

    # PCA m=8
    s, P = PCA(DTR, m=8)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    score = Poly_KernelFunction(DTR_PCA, LTR, DTE_PCA, C, constant, K, degree)
    PCA_m8_scores.append(score)

    SVM_labels = np.append(SVM_labels, LTE, axis=0)
    SVM_labels = np.hstack(SVM_labels)

    '''RAW data pi=0.1'''
    evaluation(scores_append, SVM_labels, "SVM_POLY, RAW data, ", C, K, constant, degree, 0.1)
    '''RAW data pi=0.5'''
    evaluation(scores_append, SVM_labels, "SVM_POLY, RAW data, ", C, K, constant, degree, 0.5)
    '''RAW data pi=0.9'''
    evaluation(scores_append, SVM_labels, "SVM_POLY, RAW data, ", C, K, constant, degree, 0.9)

    '''PCA with m = 9, pi=0.1'''
    evaluation(PCA_m9_scores, SVM_labels, "SVM_POLY, PCA m=9, ", C, K, constant, degree, 0.1)
    '''PCA with m = 9, pi=0.5'''
    evaluation(PCA_m9_scores, SVM_labels, "SVM_POLY, PCA m=9, ", C, K, constant, degree, 0.5)
    '''PCA with m = 9, pi=0.9'''
    evaluation(PCA_m9_scores, SVM_labels, "SVM_POLY, PCA m=9, ", C, K, constant, degree, 0.9)

    '''PCA with m = 8, pi=0.1'''
    evaluation(PCA_m8_scores, SVM_labels, "SVM_POLY, PCA m=8, ", C, K, constant, degree, 0.1)
    '''PCA with m = 8, pi=0.5'''
    evaluation(PCA_m8_scores, SVM_labels, "SVM_POLY, PCA m=8, ", C, K, constant, degree, 0.5)
    '''PCA with m = 8, pi=0.9'''
    evaluation(PCA_m8_scores, SVM_labels, "SVM_POLY, PCA m=8, ", C, K, constant, degree, 0.9)


def evaluation(scores, LR_labels, appendToTitle, C, K, constant, degree, pi):
    scores_append = np.hstack(scores)
    scores_tot = compute_dcf_min_effPrior(pi, scores_append, LR_labels)

    # plot_ROC(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C))
    # bayes_error_min_act_plot(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C), 0.4)

    t = PrettyTable(["Type", "minDCF"])
    t.title = appendToTitle + "π=" + str(pi)
    t.add_row(['SVM_POLY, K=' + str(K) + ', C=' + str(C) + ', degree=' + str(degree) + ', constant=' + str(constant),
               round(scores_tot, 3)])
    print(t)


def SVM_polynomial_tuning(DTR, LTR, DTE, LTE, C, constant, K, degree):
    scores_append = []
    PCA_m9_scores = []
    SVM_labels = []

    score = Poly_KernelFunction(DTR, LTR, DTE, C, constant, K, degree)
    scores_append.append(score)

    # PCA m=9
    s, P = PCA(DTR, m=9)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    score = Poly_KernelFunction(DTR_PCA, LTR, DTE_PCA, C, constant, K, degree)
    PCA_m9_scores.append(score)

    SVM_labels = np.append(SVM_labels, LTE, axis=0)
    SVM_labels = np.hstack(SVM_labels)

    return np.hstack(scores_append), np.hstack(PCA_m9_scores), SVM_labels
