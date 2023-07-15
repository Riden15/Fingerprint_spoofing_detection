import sys
import numpy as np

from Models.Logistic_Regression import quad_logistic_reg_score
from Models.SVM import RBF_KernelFunction
from Utility_functions.plot_validators import Roc_curve_compare, Bayes_error_plot_compare

from Models.Generative_models import *
from Models.PCA_LDA import *


def compare_evaluation_LRQ_vs_SVM_RBF(DTR, LTR, DTE, LTE):
    # lista di llr, una lista per ogni k
    labels = []

    LRQ_scores = []
    PCA_m8_SVM = []

    labels = np.append(labels, LTE, axis=0)
    labels = np.hstack(labels)
    # Once we have computed our folds, we can try different models

    ''' SCORE LRQ pi=0.5 l=0.4 '''
    def vecxxT(x):
        x = x[:, None]
        xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
        return xxT
    expanded_DTR = numpy.apply_along_axis(vecxxT, 0, DTR)
    expanded_DTE = numpy.apply_along_axis(vecxxT, 0, DTE)
    phi = numpy.vstack([expanded_DTR, DTR])
    phi_DTE = numpy.vstack([expanded_DTE, DTE])
    LRQ_scores.append(quad_logistic_reg_score(phi, LTR, phi_DTE, 0.4, 0.5))

    ''' SCORE PCA WITH 8 DIMENSIONS SVM RBF '''
    s, P = PCA(DTR, m=8)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    score = RBF_KernelFunction(DTR_PCA, LTR, DTE_PCA, 10, 0.1, 0.001)
    PCA_m8_SVM.append(score)

    LRQ_scores = np.hstack(LRQ_scores)
    PCA_m8_SVM = np.hstack(PCA_m8_SVM)
    Roc_curve_compare(LRQ_scores, PCA_m8_SVM, labels, "LRQ", "SVM_RBF", folder='evaluation/')
    Bayes_error_plot_compare(LRQ_scores, PCA_m8_SVM, labels, "LRQ", "SVM_RBF", folder='evaluation/')

