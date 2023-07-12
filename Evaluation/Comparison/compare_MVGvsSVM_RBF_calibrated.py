import sys
import numpy as np

from Models.Logistic_Regression import logistic_reg_calibration
from Models.SVM import RBF_KernelFunction
from Utility_functions.plot_validators import Roc_curve_compare, Bayes_error_plot_compare

from Models.Generative_models import *
from Models.PCA_LDA import *


def compare_evaluation_MVG_vs_SVM_RBF_calibrated(DTR, LTR, DTE, LTE):
    # lista di llr, una lista per ogni k
    labels = []

    PCA_m9_mvg = []
    PCA_m8_SVM = []

    labels = np.append(labels, LTE, axis=0)
    labels = np.hstack(labels)
    # Once we have computed our folds, we can try different models

    ''' SCORE PCA WITH 9 DIMENSIONS MVG '''
    s, P = PCA(DTR, m=9)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    PCA_m9_mvg = compute_MVG_score(DTR_PCA, LTR, DTE_PCA, PCA_m9_mvg)

    ''' SCORE PCA WITH 8 DIMENSIONS SVM RBF '''
    s, P = PCA(DTR, m=8)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    score = RBF_KernelFunction(DTR_PCA, LTR, DTE_PCA, 10, 0.1, 0.001)
    PCA_m8_SVM.append(score)

    PCA_m9_mvg = np.hstack(PCA_m9_mvg)
    PCA_m8_SVM = np.hstack(PCA_m8_SVM)
    cal_scores, cal_labels, w, b = calibrate_scores(PCA_m8_SVM, labels)
    final_score_SVM = numpy.dot(w.T, vrow(PCA_m8_SVM)) + b
    Roc_curve_compare(PCA_m9_mvg, final_score_SVM, labels, "MVG", "SVM_RBF_calibrated", folder='evaluation/')
    Bayes_error_plot_compare(PCA_m9_mvg, final_score_SVM, labels, "MVG", "SVM_RBF_calibrated", folder='evaluation/')


def calibrate_scores(scores, labels):
    scores_70 = scores[:int(len(scores) * 0.7)]
    scores_30 = scores[int(len(scores) * 0.7):]
    labels_70 = labels[:int(len(labels) * 0.7)]
    labels_30 = labels[int(len(labels) * 0.7):]

    S, estimated_w, estimated_b = logistic_reg_calibration(numpy.array([scores_70]), labels_70,
                                                           numpy.array([scores_30]), 1e-3)

    return numpy.array(S), labels_30, estimated_w, estimated_b


# Dte è il fold selezionato, Dtr è tutto il resto
def compute_MVG_score(Dtr, Ltr, Dte, MVG_res):
    llrs_MVG = MVG(Dtr, Ltr, Dte)

    MVG_res.append(llrs_MVG)
    return MVG_res
