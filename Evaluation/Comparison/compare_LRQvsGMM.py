import sys
import numpy as np

from Models.GMM import GMM_Full
from Models.Logistic_Regression import quad_logistic_reg_score
from Utility_functions.plot_validators import Roc_curve_compare, Bayes_error_plot_compare

from Models.Generative_models import *
from Models.PCA_LDA import *

def compare_evaluation_LRQ_vs_GMM(DTR, LTR, DTE, LTE):
    # lista di llr, una lista per ogni k
    labels = []

    LRQ_scores = []
    GMM_llr_fct = []

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

    ''' SCORE PCA WITH 8 DIMENSIONS SVM GMM '''
    # full-cov tied
    GMM_llr_fct = ll_GMM(DTR, LTR, DTE, GMM_llr_fct, 'tied_full', 3)

    GMM_llr_fct = np.hstack(GMM_llr_fct)
    LRQ_scores = np.hstack(LRQ_scores)
    Roc_curve_compare(LRQ_scores, GMM_llr_fct, labels, "LRQ", "GMM", folder='evaluation/')
    Bayes_error_plot_compare(LRQ_scores, GMM_llr_fct, labels, "LRQ", "GMM", folder='evaluation/')



def ll_GMM(Dtr, Ltr, Dte, llr, typeOf, comp):
    optimal_alpha = 0.1
    llr.extend(GMM_Full(Dtr, Dte, Ltr, optimal_alpha, 2 ** comp, typeOf).tolist())
    return llr
