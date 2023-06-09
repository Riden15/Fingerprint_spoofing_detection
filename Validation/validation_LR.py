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
        Dtr = []
        Ltr = []
        if i == 0:
            Dtr.append(np.hstack(FoldedData_List[i + 1:]))
            Ltr.append(np.hstack(FoldedLabel_List[i + 1:]))
        elif i == k - 1:
            Dtr.append(np.hstack(FoldedData_List[:i]))
            Ltr.append(np.hstack(FoldedLabel_List[:i]))
        else:
            Dtr.append(np.hstack(FoldedData_List[:i]))
            Dtr.append(np.hstack(FoldedData_List[i + 1:]))
            Ltr.append(np.hstack(FoldedLabel_List[:i]))
            Ltr.append(np.hstack(FoldedLabel_List[i + 1:]))

        Dtr = np.hstack(Dtr)
        Ltr = np.hstack(Ltr)

        Dte = Dtr[i]
        Lte = Ltr[i]

        # Calcolo scores con RAW data
        scores_append.append(lr_binary(Dtr, Ltr, Dte, l))

        ''' calcolo scores con PCA with m = 5'''
        s, P = PCA(Dtr, 5)
        DTR_PCA = numpy.dot(P.T,Dtr)
        DTE_PCA = numpy.dot(P.T,Dte)
        PCA_5_scores.append(lr_binary(DTR_PCA,Ltr,DTE_PCA,l))

        ''' calcolo scores PCA and LDA with m = 5'''
        P = LDA1(DTR_PCA, Ltr, 5)
        DTR_PCA_LDA = numpy.dot(P.T, DTR_PCA)
        DTE_PCA_LDA = numpy.dot(P.T, DTE_PCA)
        PCA_LDA_5_scores.append(lr_binary(DTR_PCA_LDA, Ltr, DTE_PCA_LDA, l))

        ''' calcolo scores PCA with m = 8'''
        s, P = PCA(Dtr, 8)
        DTR_PCA = numpy.dot(P.T,Dtr)
        DTE_PCA = numpy.dot(P.T,Dte)
        PCA_8_scores.append(lr_binary(DTR_PCA, Ltr, DTE_PCA, l))

        ''' calcolo scores PCA and LDA with m = 8'''
        P = LDA1(DTR_PCA, Ltr, 8)
        DTR_PCA_LDA = numpy.dot(P.T, DTR_PCA)
        DTE_PCA_LDA = numpy.dot(P.T, DTE_PCA)
        PCA_LDA_8_scores.append(lr_binary(DTR_PCA_LDA, Ltr, DTE_PCA_LDA, l))

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)

    '''RAW data pi=0.1'''
    evaluation(scores_append, LR_labels, appendToTitle + "RAW data ", l,C, 0.1)
    '''RAW data pi=0.5'''
    evaluation(scores_append, LR_labels, appendToTitle + "RAW data ", l, C, 0.5)
    '''RAW data pi=0.9'''
    evaluation(scores_append, LR_labels, appendToTitle + "RAW data ", l, C, 0.9)


def kFold_LR_calibration(DTR, LTR, l, k):
    FoldedData_List = numpy.split(DTR, k, axis=1)
    FoldedLabel_List = numpy.split(LTR, k)

    scores_append = []
    LR_labels = []

    for i in range(k):
        Dtr = []
        Ltr = []
        if i == 0:
            Dtr.append(np.hstack(FoldedData_List[i + 1:]))
            Ltr.append(np.hstack(FoldedLabel_List[i + 1:]))
        elif i == k - 1:
            Dtr.append(np.hstack(FoldedData_List[:i]))
            Ltr.append(np.hstack(FoldedLabel_List[:i]))
        else:
            Dtr.append(np.hstack(FoldedData_List[:i]))
            Dtr.append(np.hstack(FoldedData_List[i + 1:]))
            Ltr.append(np.hstack(FoldedLabel_List[:i]))
            Ltr.append(np.hstack(FoldedLabel_List[i + 1:]))

        Dtr = np.hstack(Dtr)
        Ltr = np.hstack(Ltr)

        Dte = FoldedData_List[i]
        Lte = FoldedLabel_List[i]

        scores = lr_binary(Dtr, Ltr, Dte, l)
        scores_append.append(scores)

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)

    return np.hstack(scores_append), LR_labels


def validation_LR(DTR, LTR, L, appendToTitle, k):
    C = np.array([[0, 1], [10, 0]])  # costi Cfp = 10, Cfn = 1
    for l in L:
        kFold_LR(DTR, LTR, l, appendToTitle, C, k)


'''
 algoritmo che serve per trovare il miglior hyper parameter L

    x = numpy.logspace(-5, 1, 20) 
    y = numpy.array([])
    y_05 = numpy.array([])
    y_09 = numpy.array([])
    y_01 = numpy.array([])

    for xi in x:
        scores, labels = kFold_LR_calibration(DTR, LTR, xi, k)
        y_05 = numpy.hstack((y_05, compute_dcf_min(0.5, C, scores, labels)))
        y_09 = numpy.hstack((y_09, compute_dcf_min(0.9, C, scores, labels)))
        y_01 = numpy.hstack((y_01, compute_dcf_min(0.1, C, scores, labels)))

    y = numpy.hstack((y, y_05))
    y = numpy.vstack((y, y_09))
    y = numpy.vstack((y, y_01))

    plot_DCF(x, y, 'lambda', appendToTitle + 'LR_minDCF_comparison')
'''
