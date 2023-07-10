# -*- coding: utf-8 -*-

import numpy as np

from Utility_functions.plot_validators import *
from Models.Logistic_Regression import *
from prettytable import PrettyTable
from Models.PCA_LDA import *


def evaluation_LR(DTR, LTR, DTE, LTE, l):
    evaluate_LR(DTR, LTR, DTE, LTE, l)

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

def evaluate_LR(DTR, LTR, DTE, LTE, l):
    scores_append = []
    PCA_m9_scores = []
    PCA_m8_scores = []
    LR_labels = []

    # Calcolo scores con RAW data
    scores_append.append(lr_binary(DTR, LTR, DTE, l))

    ''' calcolo scores con PCA with m = 9'''
    s, P = PCA(DTR, 9)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    PCA_m9_scores.append(lr_binary(DTR_PCA, LTR, DTE_PCA, l))

    ''' calcolo scores PCA with m = 8'''
    s, P = PCA(DTR, 8)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    PCA_m8_scores.append(lr_binary(DTR_PCA, LTR, DTE_PCA, l))

    LR_labels = np.append(LR_labels, LTE, axis=0)
    LR_labels = np.hstack(LR_labels)

    '''RAW data pi=0.1'''
    evaluation(scores_append, LR_labels, "LR, RAW data ", l, 0.1)
    '''RAW data pi=0.5'''
    evaluation(scores_append, LR_labels, "LR, RAW data ", l, 0.5)
    '''RAW data pi=0.9'''
    evaluation(scores_append, LR_labels, "LR, RAW data ", l, 0.9)

    '''PCA with m = 9, pi=0.1'''
    evaluation(PCA_m9_scores, LR_labels, "LR, PCA m=9, ", l, 0.1)
    '''PCA with m = 9, pi=0.5'''
    evaluation(PCA_m9_scores, LR_labels, "LR, PCA m=9, ", l, 0.5)
    '''PCA with m = 9, pi=0.9'''
    evaluation(PCA_m9_scores, LR_labels, "LR, PCA m=9, ", l, 0.9)

    '''PCA with m  = 8, pi=0.1'''
    evaluation(PCA_m8_scores, LR_labels, "LR, PCA m=8, ", l, 0.1)
    '''PCA with m  = 8 pi=0.5'''
    evaluation(PCA_m8_scores, LR_labels, "LR, PCA m=8, ", l, 0.5)
    '''PCA with m  = 8 pi=0.9'''
    evaluation(PCA_m8_scores, LR_labels, "LR, PCA m=8, ", l, 0.9)


def evaluation(scores, LR_labels, appendToTitle, l, pi):
    scores_append = np.hstack(scores)
    scores_tot = compute_dcf_min_effPrior(pi, scores_append, LR_labels)

    # Roc_curve(scores_append, LR_labels, appendToTitle + 'LR, lambda=' + str(l))

    t = PrettyTable(["lamda", "minDCF"])
    t.title = appendToTitle + "π=" + str(pi)
    t.add_row([str(l), round(scores_tot, 3)])
    print(t)


def LR_tuning(DTR, LTR, DTE, LTE, l):
    scores_append = []
    PCA_m9_scores = []
    LR_labels = []

    # Calcolo scores con RAW data
    scores_append.append(lr_binary(DTR, LTR, DTE, l))

    ''' calcolo scores con PCA with m = 9'''
    s, P = PCA(DTR, 9)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    PCA_m9_scores.append(lr_binary(DTR_PCA, LTR, DTE_PCA, l))

    LR_labels = np.append(LR_labels, LTE, axis=0)
    LR_labels = np.hstack(LR_labels)

    return np.hstack(scores_append), np.hstack(PCA_m9_scores), LR_labels
