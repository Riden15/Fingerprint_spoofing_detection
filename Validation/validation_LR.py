# -*- coding: utf-8 -*-

import numpy as np

from Validators import *
from Models.Logistic_Regression import *
from prettytable import PrettyTable

def evaluation(scores, LR_labels, appendToTitle, l, C, pi):
    scores_append = np.hstack(scores)
    scores_tot = compute_dcf_min(pi,C,scores_append, LR_labels)

    #Roc_curve(scores_append, LR_labels, appendToTitle + 'LR, lambda=' + str(l))

    t = PrettyTable(["Type", "minDCF"])
    t.title = appendToTitle + "minDCF: π=0.5"
    t.add_row(['LR, lambda=' + str(l), round(scores_tot, 3)])
    print(t)

    ###############################

    # π = 0.1
    scores_tot = compute_dcf_min(0.1,C,scores_append, LR_labels)

    t = PrettyTable(["Type", "minDCF"])
    t.title = appendToTitle + "minDCF: π=0.1"
    t.add_row(['LR, lambda=' + str(l), round(scores_tot, 3)])

    print(t)

    ###############################

    # π = 0.9
    scores_tot = compute_dcf_min(0.9,C,scores_append, LR_labels)

    t = PrettyTable(["Type", "minDCF"])
    t.title = appendToTitle + "minDCF: π=0.9"
    t.add_row(['LR, lambda=' + str(l), round(scores_tot, 3)])

    print(t)


def kfold_LR(DTR, LTR, l, appendToTitle, C):
    k = 5
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)

    scores_append = []

    LR_labels = []

    for i in range(k):
        D = []
        L = []
        if i == 0:
            D.append(np.hstack(Dtr[i + 1:]))
            L.append(np.hstack(Ltr[i + 1:]))
        elif i == k - 1:
            D.append(np.hstack(Dtr[:i]))
            L.append(np.hstack(Ltr[:i]))
        else:
            D.append(np.hstack(Dtr[:i]))
            D.append(np.hstack(Dtr[i + 1:]))
            L.append(np.hstack(Ltr[:i]))
            L.append(np.hstack(Ltr[i + 1:]))

        D = np.hstack(D)
        L = np.hstack(L)

        Dte = Dtr[i]
        Lte = Ltr[i]

        # Calcolo scores con RAW data
        scores_append.append(lr_binary(D, L, Dte, l))

        ''' calcolo scores con PCA with m = 5'''
        P = PCA(D, 5)
        DTR_5_PCA = numpy.dot(P.T,D)
        DTE_5_PCA = numpy.dot(P.T,Dte)
        PCA_5_scores.append(lr_binary(DTR_5_PCA,L,DTE_5_PCA,l))

        ''' calcolo scores PCA with m = 8'''
        P = PCA(D, 8)
        DTR_8_PCA = numpy.dot(P.T,D)
        DTE_8_PCA = numpy.dot(P.T,Dte)
        PCA_8_scores.append(lr_binary(DTR_8_PCA, L, DTE_8_PCA, l))

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)

    '''RAW data pi=0.1'''
    evaluation(scores_append, LR_labels, appendToTitle + "RAW data ", l,C, 0.1)
    '''RAW data pi=0.5'''
    evaluation(scores_append, LR_labels, appendToTitle + "RAW data ", l, C, 0.5)
    '''RAW data pi=0.9'''
    evaluation(scores_append, LR_labels, appendToTitle + "RAW data ", l, C, 0.9)


def kfold_LR_calibration(DTR, LTR, l):
    k = 15
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)

    scores_append = []
    LR_labels = []

    for i in range(k):
        D = []
        L = []
        if i == 0:
            D.append(np.hstack(Dtr[i + 1:]))
            L.append(np.hstack(Ltr[i + 1:]))
        elif i == k - 1:
            D.append(np.hstack(Dtr[:i]))
            L.append(np.hstack(Ltr[:i]))
        else:
            D.append(np.hstack(Dtr[:i]))
            D.append(np.hstack(Dtr[i + 1:]))
            L.append(np.hstack(Ltr[:i]))
            L.append(np.hstack(Ltr[i + 1:]))

        D = np.hstack(D)
        L = np.hstack(L)

        Dte = Dtr[i]
        Lte = Ltr[i]


        scores = lr_binary(D, L, Dte, l)
        scores_append.append(scores)

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)

    return np.hstack(scores_append), LR_labels


def validation_LR(DTR, LTR, L, appendToTitle, k):
    C = np.array([[0, 1], [10, 0]])  # costi Cfp = 10, Cfn = 1
    for l in L:
        kfold_LR(DTR, LTR, l, appendToTitle, C, k)

#todo controllare se va messa la effective prior o se la C va cambiata
    x = numpy.logspace(-5, 1, 5)
    y = numpy.array([])
    y_05 = numpy.array([])
    y_09 = numpy.array([])
    y_01 = numpy.array([])

    for xi in x:
        scores, labels = kfold_LR_calibration(DTR, LTR, xi)
        y_05 = numpy.hstack((y_05, Bayes_error_plot(0.5, scores, labels)))
        y_09 = numpy.hstack((y_09, Bayes_error_plot(0.9, scores, labels)))
        y_01 = numpy.hstack((y_01, Bayes_error_plot(0.1, scores, labels)))

    y = numpy.hstack((y, y_05))
    y = numpy.vstack((y, y_09))
    y = numpy.vstack((y, y_01))

    plot_DCF(x, y, 'lambda', appendToTitle + 'LR_minDCF_comparison')
