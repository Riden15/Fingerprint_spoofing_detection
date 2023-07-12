import sys
import numpy as np

from Models.GMM import GMM_Full
from Models.Logistic_Regression import logistic_reg_calibration
from Models.SVM import RBF_KernelFunction
from Utility_functions.plot_validators import Roc_curve_compare, Bayes_error_plot_compare
from Models.PCA_LDA import *


def compare_validation_GMM_vs_SVM_RBF_calibrated(DTR, LTR, k):
    FoldedData_List = np.split(DTR, k, axis=1)  # lista di fold
    FoldedLabel_List = np.split(LTR, k)

    # lista di llr, una lista per ogni k
    labels = []

    GMM_llr_fct = []
    PCA_m8_SVM = []

    for fold in range(k):
        Dtr, Ltr, Dte, Lte = kfold(fold, k, FoldedData_List, FoldedLabel_List)

        labels = np.append(labels, Lte, axis=0)
        labels = np.hstack(labels)
        # Once we have computed our folds, we can try different models

        ''' SCORE PCA WITH 8 DIMENSIONS SVM GMM '''
        # full-cov tied
        GMM_llr_fct = ll_GMM(Dtr, Ltr, Dte, GMM_llr_fct, 'tied_full', 3)

        ''' SCORE PCA WITH 8 DIMENSIONS SVM RBF '''
        s, P = PCA(Dtr, m=8)
        DTR_PCA = numpy.dot(P.T, Dtr)
        DTE_PCA = numpy.dot(P.T, Dte)
        score = RBF_KernelFunction(DTR_PCA, Ltr, DTE_PCA, 10, 0.1, 0.001)
        PCA_m8_SVM.append(score)

    GMM_llr_fct = np.hstack(GMM_llr_fct)
    PCA_m8_SVM = np.hstack(PCA_m8_SVM)
    cal_scores, cal_labels, w, b = calibrate_scores(PCA_m8_SVM, labels)
    final_score_SVM = numpy.dot(w.T, vrow(PCA_m8_SVM)) + b
    Roc_curve_compare(GMM_llr_fct, final_score_SVM, labels, "GMM", "SVM_RBF_calibrated", folder='validation/')
    Bayes_error_plot_compare(GMM_llr_fct, final_score_SVM, labels, "GMM", "SVM_RBF_calibrated", folder='validation/')


def calibrate_scores(scores, labels):
    scores_70 = scores[:int(len(scores) * 0.7)]
    scores_30 = scores[int(len(scores) * 0.7):]
    labels_70 = labels[:int(len(labels) * 0.7)]
    labels_30 = labels[int(len(labels) * 0.7):]

    S, estimated_w, estimated_b = logistic_reg_calibration(numpy.array([scores_70]), labels_70,
                                                           numpy.array([scores_30]), 1e-3)

    return numpy.array(S), labels_30, estimated_w, estimated_b


def ll_GMM(Dtr, Ltr, Dte, llr, typeOf, comp):
    optimal_alpha = 0.1
    llr.extend(GMM_Full(Dtr, Dte, Ltr, optimal_alpha, 2 ** comp, typeOf).tolist())
    return llr
