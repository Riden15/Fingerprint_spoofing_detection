import sys
import numpy as np

from Models.GMM import GMM_Full
from Utility_functions.plot_validators import Roc_curve_compare, Bayes_error_plot_compare

from Models.Generative_models import *
from Models.PCA_LDA import *


def compare_evaluation_MVG_vs_GMM(DTR, LTR, DTE, LTE):
    # lista di llr, una lista per ogni k
    labels = []

    PCA_m9_mvg = []
    GMM_llr_fct = []

    labels = np.append(labels, LTE, axis=0)
    labels = np.hstack(labels)
    # Once we have computed our folds, we can try different models

    ''' SCORE PCA WITH 9 DIMENSIONS MVG '''
    s, P = PCA(DTR, m=9)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    PCA_m9_mvg = compute_MVG_score(DTR_PCA, LTR, DTE_PCA, PCA_m9_mvg)

    ''' SCORE PCA WITH 8 DIMENSIONS SVM GMM '''
    # full-cov tied
    GMM_llr_fct = ll_GMM(DTR, LTR, DTE, GMM_llr_fct, 'tied_full', 3)

    GMM_llr_fct = np.hstack(GMM_llr_fct)
    PCA_m9_mvg = np.hstack(PCA_m9_mvg)
    Roc_curve_compare(PCA_m9_mvg, GMM_llr_fct, labels, "MVG", "GMM", folder='evaluation/')
    Bayes_error_plot_compare(PCA_m9_mvg, GMM_llr_fct, labels, "MVG", "GMM", folder='evaluation/')


# Dte è il fold selezionato, Dtr è tutto il resto
def compute_MVG_score(Dtr, Ltr, Dte, MVG_res):
    llrs_MVG = MVG(Dtr, Ltr, Dte)

    MVG_res.append(llrs_MVG)
    return MVG_res


def ll_GMM(Dtr, Ltr, Dte, llr, typeOf, comp):
    optimal_alpha = 0.1
    llr.extend(GMM_Full(Dtr, Dte, Ltr, optimal_alpha, 2 ** comp, typeOf).tolist())
    return llr
