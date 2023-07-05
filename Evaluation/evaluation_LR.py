# -*- coding: utf-8 -*-

import numpy as np

from Utility_functions.plot_validators import *
from Models.Logistic_Regression import *
from prettytable import PrettyTable
from Models.PCA_LDA import *

def evaluation_LR(DTR, LTR, DTE, LTE, l):

    scores_append = []
    PCA_m9_scores = []
    PCA_m8_scores = []
    LR_labels = []

    # Calcolo scores con RAW data
    scores_append.append(lr_binary(DTR, LTR, DTE, l))

    ''' calcolo scores con PCA with m = 9'''
    s, P = PCA(DTR, 9)
    DTR_PCA = numpy.dot(P.T,DTR)
    DTE_PCA = numpy.dot(P.T,DTE)
    PCA_m9_scores.append(lr_binary(DTR_PCA,LTR,DTE_PCA,l))

    ''' calcolo scores PCA with m = 8'''
    s, P = PCA(DTR, 8)
    DTR_PCA = numpy.dot(P.T,DTR)
    DTE_PCA = numpy.dot(P.T,DTE)
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

    #Roc_curve(scores_append, LR_labels, appendToTitle + 'LR, lambda=' + str(l))

    t = PrettyTable(["lamda", "minDCF"])
    t.title = appendToTitle + "π=" + str(pi)
    t.add_row([str(l), round(scores_tot, 3)])
    print(t)