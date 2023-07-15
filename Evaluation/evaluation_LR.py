# -*- coding: utf-8 -*-

import numpy as np

from Utility_functions.plot_validators import *
from Models.Logistic_Regression import *
from prettytable import PrettyTable
from Models.PCA_LDA import *


def evaluation_LR(DTR, LTR, DTE, LTE, l):
    for pi in [0.1, 0.5, 0.9]:
        evaluate_LR(DTR, LTR, DTE, LTE, l, pi)

'''
    x = numpy.logspace(-5, 1, 20)
    y = numpy.array([])
    y_05 = numpy.array([])
    y_09 = numpy.array([])
    y_01 = numpy.array([])
    y_05_PCA = numpy.array([])
    y_09_PCA = numpy.array([])
    y_01_PCA = numpy.array([])

    for xi in x:
        scores, scoresPCA, labels = LR_tuning(DTR, LTR, DTE, LTE, xi)
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

    plot_DCF_PCA(x, y, 'lambda', 'LR_PCA_minDCF_comparison', folder='evaluation/')
'''

def evaluate_LR(DTR, LTR, DTE, LTE, l, pi):
    scores_append = []
    PCA_m9_scores = []
    PCA_m8_scores = []
    LR_labels = []

    # Calcolo scores con RAW data
    scores_append.append(weighted_logistic_reg_score(DTR, LTR, DTE, l, pi))

    ''' calcolo scores con PCA with m = 9'''
    s, P = PCA(DTR, 9)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    PCA_m9_scores.append(weighted_logistic_reg_score(DTR_PCA, LTR, DTE_PCA, l, pi))

    ''' calcolo scores PCA with m = 8'''
    s, P = PCA(DTR, 8)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    PCA_m8_scores.append(weighted_logistic_reg_score(DTR_PCA, LTR, DTE_PCA, l, pi))

    LR_labels = np.append(LR_labels, LTE, axis=0)
    LR_labels = np.hstack(LR_labels)

    '''RAW data pi=0.1'''
    evaluation(scores_append, LR_labels, "LR, RAW data ", l, pi)
    '''PCA with m = 9'''
    evaluation(PCA_m9_scores, LR_labels, "LR, PCA m=9, ", l, pi)
    '''PCA with m  = 8'''
    evaluation(PCA_m8_scores, LR_labels, "LR, PCA m=8, ", l, pi)


def evaluation(scores, LR_labels, appendToTitle, l, pi):
    scores_append = np.hstack(scores)
    scores_tot_01 = compute_dcf_min_effPrior(0.1, scores_append, LR_labels)
    scores_tot_05 = compute_dcf_min_effPrior(0.5, scores_append, LR_labels)
    scores_tot_09 = compute_dcf_min_effPrior(0.9, scores_append, LR_labels)

    t = PrettyTable(["Type", "π=0.5", "π=0.1", "π=0.9"])
    t.title = appendToTitle
    t.add_row(['WEIGHTED_LR, lambda=' + str(l) + " π_t=" + str(pi), round(scores_tot_05, 3), round(scores_tot_01, 3), round(scores_tot_09, 3)])
    print(t)


def LR_tuning(DTR, LTR, DTE, LTE, l):
    scores_append = []
    PCA_m9_scores = []
    LR_labels = []

    # Calcolo scores con RAW data
    scores_append.append(weighted_logistic_reg_score(DTR, LTR, DTE, l, 0.5))

    ''' calcolo scores con PCA with m = 9'''
    s, P = PCA(DTR, 9)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    PCA_m9_scores.append(weighted_logistic_reg_score(DTR_PCA, LTR, DTE_PCA, l, 0.5))

    LR_labels = np.append(LR_labels, LTE, axis=0)
    LR_labels = np.hstack(LR_labels)

    return np.hstack(scores_append), np.hstack(PCA_m9_scores), LR_labels
