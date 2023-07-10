import sys
import numpy as np
from prettytable import PrettyTable

sys.path.append('../')
from Utility_functions.Validators import *
from Models.SVM import *
from Models.PCA_LDA import *


def validation_SVM_RBF(DTR, LTR, K_arr, gamma_arr, C_arr, k):
    for C in C_arr:
        for K in K_arr:
            for gamma in gamma_arr:
                kfold_SVM_RBF(DTR, LTR, C, K, gamma, k)


'''
    x = numpy.logspace(-3, 3, 15)  # x contains different values of C
    y = numpy.array([])
    gamma_minus_4 = numpy.array([])
    gamma_minus_3 = numpy.array([])
    gamma_minus_2 = numpy.array([])
    gamma_minus_1 = numpy.array([])

    for xi in x:                                                        #  C    K    g
        scores_gamma_minus_4, labels = kfold_svm_rbf_calibration(DTR, LTR, xi, 1.0, 1e-4, k)
        scores_gamma_minus_3, _ = kfold_svm_rbf_calibration(DTR, LTR, xi, 1.0, 1e-3, k)
        scores_gamma_minus_2, _ = kfold_svm_rbf_calibration(DTR, LTR, xi, 1.0, 1e-2, k)
        scores_gamma_minus_1, _ = kfold_svm_rbf_calibration(DTR, LTR, xi, 1.0, 1e-1, k)

        gamma_minus_4 = numpy.hstack((gamma_minus_4, compute_dcf_min_effPrior(0.5, scores_gamma_minus_4, labels)))
        gamma_minus_3 = numpy.hstack((gamma_minus_3, compute_dcf_min_effPrior(0.5, scores_gamma_minus_3, labels)))
        gamma_minus_2 = numpy.hstack((gamma_minus_2, compute_dcf_min_effPrior(0.5, scores_gamma_minus_2, labels)))
        gamma_minus_1 = numpy.hstack((gamma_minus_1, compute_dcf_min_effPrior(0.5, scores_gamma_minus_1, labels)))

    y = numpy.hstack((y, gamma_minus_4))
    y = numpy.vstack((y, gamma_minus_3))
    y = numpy.vstack((y, gamma_minus_2))
    y = numpy.vstack((y, gamma_minus_1))

    plot_DCF_for_SVM_RBF_calibration(x, y, 'C', 'SVM_RBF_minDCF_comparison', folder='validation/')
'''


def kfold_SVM_RBF(DTR, LTR, C, K, gamma, k):
    FoldedData_List = numpy.split(DTR, k, axis=1)
    FoldedLabel_List = numpy.split(LTR, k)

    scores_append = []
    PCA_m9_scores = []
    PCA_m8_scores = []
    SVM_labels = []

    for fold in range(k):
        Dtr, Ltr, Dte, Lte = kfold(fold, k, FoldedData_List, FoldedLabel_List)

        score = RBF_KernelFunction(Dtr, Ltr, Dte, C, K, gamma)
        scores_append.append(score)

        # PCA m=9
        s, P = PCA(Dtr, m=9)
        DTR_PCA = numpy.dot(P.T, Dtr)
        DTE_PCA = numpy.dot(P.T, Dte)
        score = RBF_KernelFunction(DTR_PCA, Ltr, DTE_PCA, C, K, gamma)
        PCA_m9_scores.append(score)

        # PCA m=8
        s, P = PCA(Dtr, m=8)
        DTR_PCA = numpy.dot(P.T, Dtr)
        DTE_PCA = numpy.dot(P.T, Dte)
        score = RBF_KernelFunction(DTR_PCA, Ltr, DTE_PCA, C, K, gamma)
        PCA_m8_scores.append(score)

        SVM_labels = np.append(SVM_labels, Lte, axis=0)
        SVM_labels = np.hstack(SVM_labels)

    '''RAW data pi=0.1'''
    evaluation(scores_append, SVM_labels, "SVM_RBF, RAW data, ", C, K, gamma, 0.1)
    '''RAW data pi=0.5'''
    evaluation(scores_append, SVM_labels, "SVM_RBF, RAW data, ", C, K, gamma, 0.5)
    '''RAW data pi=0.9'''
    evaluation(scores_append, SVM_labels, "SVM_RBF, RAW data, ", C, K, gamma, 0.9)

    '''PCA with m = 9, pi=0.1'''
    evaluation(PCA_m9_scores, SVM_labels, "SVM_RBF, PCA m=9, ", C, K, gamma, 0.1)
    '''PCA with m = 9, pi=0.5'''
    evaluation(PCA_m9_scores, SVM_labels, "SVM_RBF, PCA m=9, ", C, K, gamma, 0.5)
    '''PCA with m = 9, pi=0.9'''
    evaluation(PCA_m9_scores, SVM_labels, "SVM_RBF, PCA m=9, ", C, K, gamma, 0.9)

    '''PCA with m = 8, pi=0.1'''
    evaluation(PCA_m8_scores, SVM_labels, "SVM_RBF, PCA m=8, ", C, K, gamma, 0.1)
    '''PCA with m = 8, pi=0.5'''
    evaluation(PCA_m8_scores, SVM_labels, "SVM_RBF, PCA m=8, ", C, K, gamma, 0.5)
    '''PCA with m = 8, pi=0.9'''
    evaluation(PCA_m8_scores, SVM_labels, "SVM_RBF, PCA m=8, ", C, K, gamma, 0.9)


def evaluation(scores, LR_labels, appendToTitle, C, K, gamma, pi):
    scores_append = np.hstack(scores)
    scores_tot = compute_dcf_min_effPrior(pi, scores_append, LR_labels)

    # act_DCF_05 = compute_act_DCF(scores_append, SVM_labels, 0.5, 1, 1, )
    # plot_ROC(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C))
    # bayes_error_min_act_plot(scores_append, SVM_labels, appendToTitle + 'SVM_RFB, K=' + str(K) + ', C=' + str(C), 0.4)

    t = PrettyTable(["Type", "minDCF"])
    t.title = appendToTitle + "π=" + str(pi)
    t.add_row(['SVM_RBF, K=' + str(K) + ', C=' + str(C) + ', gamma=' + str(gamma), round(scores_tot, 3)])
    print(t)


def kfold_svm_rbf_calibration(DTR, LTR, C, K, gamma, k):
    FoldedData_List = numpy.split(DTR, k, axis=1)
    FoldedLabel_List = numpy.split(LTR, k)

    scores_append = []
    SVM_labels = []

    for fold in range(k):
        Dtr, Ltr, Dte, Lte = kfold(fold, k, FoldedData_List, FoldedLabel_List)

        score = RBF_KernelFunction(Dtr, Ltr, Dte, C, K, gamma)
        scores_append.append(score)

        SVM_labels = np.append(SVM_labels, Lte, axis=0)
        SVM_labels = np.hstack(SVM_labels)

    return np.hstack(scores_append), SVM_labels
