import sys
import numpy as np
from prettytable import PrettyTable

sys.path.append('../')
from Utility_functions.Validators import *
from Models.SVM import *
from Models.PCA_LDA import *


def evaluation_SVM_RBF(DTR, LTR, DTE, LTE, K, gamma, C):
    scores_append = []
    PCA_m9_scores = []
    PCA_m8_scores = []
    SVM_labels = []

    score = RBF_KernelFunction(DTR, LTR, DTE, C, K, gamma)
    scores_append.append(score)

    # PCA m=9
    s, P = PCA(DTR, m=9)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    score = RBF_KernelFunction(DTR_PCA, LTR, DTE_PCA, C, K, gamma)
    PCA_m9_scores.append(score)

    # PCA m=8
    s, P = PCA(DTR, m=8)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    score = RBF_KernelFunction(DTR_PCA, LTR, DTE_PCA, C, K, gamma)
    PCA_m8_scores.append(score)

    SVM_labels = np.append(SVM_labels, LTE, axis=0)
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
