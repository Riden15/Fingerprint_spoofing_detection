import sys
import numpy as np

from Models.GMM import GMM_Full
from Models.Logistic_Regression import quad_logistic_reg_score
from Models.SVM import RBF_KernelFunction
from Utility_functions.plot_validators import Roc_curve_compare, Bayes_error_plot_compare

from Models.Generative_models import *
from Utility_functions.Validators import *
from prettytable import PrettyTable
from Models.PCA_LDA import *

def compare_validation_LRQ_vs_GMM(DTR, LTR, k):
    FoldedData_List = np.split(DTR, k, axis=1)  # lista di fold
    FoldedLabel_List = np.split(LTR, k)

    # lista di llr, una lista per ogni k
    labels = []

    LRQ_scores = []
    GMM_llr_fct = []

    for fold in range(k):
        Dtr, Ltr, Dte, Lte = kfold(fold, k, FoldedData_List, FoldedLabel_List)

        labels = np.append(labels, Lte, axis=0)
        labels = np.hstack(labels)

        ''' SCORE LRQ pi=0.5 l=0.4 '''
        def vecxxT(x):
            x = x[:, None]
            xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
            return xxT
        expanded_DTR = numpy.apply_along_axis(vecxxT, 0, Dtr)
        expanded_DTE = numpy.apply_along_axis(vecxxT, 0, Dte)
        phi = numpy.vstack([expanded_DTR, Dtr])
        phi_DTE = numpy.vstack([expanded_DTE, Dte])
        LRQ_scores.append( quad_logistic_reg_score(phi, Ltr, phi_DTE, 0.4, 0.5))

        ''' SCORE PCA WITH 8 DIMENSIONS SVM GMM '''
        # full-cov tied
        GMM_llr_fct = ll_GMM(Dtr, Ltr, Dte, GMM_llr_fct, 'tied_full', 3)

    GMM_llr_fct = np.hstack(GMM_llr_fct)
    LRQ_scores = np.hstack(LRQ_scores)
    Roc_curve_compare(LRQ_scores, GMM_llr_fct, labels, "LRQ", "GMM", folder='validation/')
    Bayes_error_plot_compare(LRQ_scores, GMM_llr_fct, labels, "LRQ", "GMM", folder='validation/')


def ll_GMM(Dtr, Ltr, Dte, llr, typeOf, comp):
    optimal_alpha = 0.1
    llr.extend(GMM_Full(Dtr, Dte, Ltr, optimal_alpha, 2 ** comp, typeOf).tolist())
    return llr