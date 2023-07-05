import sys
import numpy as np

sys.path.append('../')
from Models.SVM import *
from Utility_functions.Validators import *
from prettytable import PrettyTable
from Models.PCA_LDA import *


def evaluation_SVM(DTR, LTR, DTE, LTE, K, C):
    scores_append = []
    PCA_m9_scores = []
    SVM_labels = []

    wStar, primal = train_SVM_linear(DTR, LTR, C=C, K=K)
    DTEEXT = numpy.vstack([DTE, K * numpy.ones((1, DTE.shape[1]))])
    scores = numpy.dot(wStar.T, DTEEXT).ravel()
    scores_append.append(scores)

    # PCA m=9
    s, P = PCA(DTR, m=9)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)

    wStar, primal = train_SVM_linear(DTR_PCA, LTR, C=C, K=K)
    DTEEXT = numpy.vstack([DTE_PCA, K * numpy.ones((1, DTE.shape[1]))])
    PCA2_SVM_scores = numpy.dot(wStar.T, DTEEXT).ravel()
    PCA_m9_scores.append(PCA2_SVM_scores)

    SVM_labels = np.append(SVM_labels, LTE, axis=0)
    SVM_labels = np.hstack(SVM_labels)

    '''RAW data pi=0.1'''
    evaluation(scores_append, SVM_labels, "SVM, RAW data, ", C, K, 0.1)
    '''RAW data pi=0.5'''
    evaluation(scores_append, SVM_labels, "SVM, RAW data, ", C, K, 0.5)
    '''RAW data pi=0.9'''
    evaluation(scores_append, SVM_labels, "SVM, RAW data, ", C, K, 0.9)

    '''PCA with m = 9, pi=0.1'''
    evaluation(PCA_m9_scores, SVM_labels, "SVM, PCA m=9, ", C, K, 0.1)
    '''PCA with m = 9, pi=0.5'''
    evaluation(PCA_m9_scores, SVM_labels, "SVM, PCA m=9, ", C, K, 0.5)
    '''PCA with m = 9, pi=0.9'''
    evaluation(PCA_m9_scores, SVM_labels, "SVM, PCA m=9, ", C, K, 0.9)


def evaluation(scores, LR_labels, appendToTitle, C, K, pi):
    scores_append = np.hstack(scores)
    scores_tot = compute_dcf_min_effPrior(pi, scores_append, LR_labels)
    # plot_ROC(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C))

    # Cfn and Ctp are set to 1
    # bayes_error_min_act_plot(scores_append, SVM_labels, appendToTitle + 'SVM, K=' + str(K) + ', C=' + str(C), 0.4)

    t = PrettyTable(["Type", "minDCF"])
    t.title = appendToTitle + "π=" + str(pi)
    t.add_row(['SVM, K=' + str(K) + ', C=' + str(C), round(scores_tot, 3)])
    print(t)
