import sys

import numpy

from Models.Logistic_Regression import logistic_reg_calibration
from Models.PCA_LDA import *
from Models.SVM import RBF_KernelFunction
from Utility_functions.plot_validators import Bayes_error_plot


def calibrate_scores(scores, labels):
    scores_70 = scores[:int(len(scores) * 0.7)]
    scores_30 = scores[int(len(scores) * 0.7):]
    labels_70 = labels[:int(len(labels) * 0.7)]
    labels_30 = labels[int(len(labels) * 0.7):]

    S, estimated_w, estimated_b = logistic_reg_calibration(numpy.array([scores_70]), labels_70,
                                                           numpy.array([scores_30]), 1e-3)

    return numpy.array(S), labels_30, estimated_w, estimated_b


def evaluation_SVM_RBF_score_calibration(DTR, LTR, DTE, LTE):
    PCA_m8_SVM = []
    SVM_labels = []

    SVM_labels = numpy.append(SVM_labels, LTE, axis=0)
    SVM_labels = numpy.hstack(SVM_labels)

    ''' SCORE PCA WITH 8 DIMENSIONS SVM RBF '''
    s, P = PCA(DTR, m=8)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    score = RBF_KernelFunction(DTR_PCA, LTR, DTE_PCA, 10, 0.1, 0.001)
    PCA_m8_SVM.append(score)

    PCA_m8_SVM = numpy.hstack(PCA_m8_SVM)
    cal_scores, cal_labels, w, b = calibrate_scores(PCA_m8_SVM, SVM_labels)
    final_score = numpy.dot(w.T, vrow(PCA_m8_SVM)) + b

    Bayes_error_plot(final_score, SVM_labels, "SVM_RBF_calibrated", folder='evaluation/')
