import sys
import numpy as np

from Models.GMM import GMM_Full
from Models.SVM import RBF_KernelFunction
from Utility_functions.plot_validators import Roc_curve_compare, Bayes_error_plot_compare

from Models.Generative_models import *
from Utility_functions.Validators import *
from prettytable import PrettyTable
from Models.PCA_LDA import *

def compare_validation_MVG_vs_GMM(DTR, LTR, k):
    FoldedData_List = np.split(DTR, k, axis=1)  # lista di fold
    FoldedLabel_List = np.split(LTR, k)

    # lista di llr, una lista per ogni k
    labels = []

    PCA_m9_mvg = []
    GMM_llr_fct = []

    for fold in range(k):
        Dtr, Ltr, Dte, Lte = kfold(fold, k, FoldedData_List, FoldedLabel_List)

        labels = np.append(labels, Lte, axis=0)
        labels = np.hstack(labels)
        # Once we have computed our folds, we can try different models

        ''' SCORE PCA WITH 9 DIMENSIONS MVG '''
        s, P = PCA(Dtr, m=9)
        DTR_PCA = numpy.dot(P.T, Dtr)
        DTE_PCA = numpy.dot(P.T, Dte)
        PCA_m9_mvg = compute_MVG_score(DTR_PCA, Ltr, DTE_PCA, PCA_m9_mvg)

        ''' SCORE PCA WITH 8 DIMENSIONS SVM GMM '''
        # full-cov tied
        GMM_llr_fct = ll_GMM(Dtr, Ltr, Dte, GMM_llr_fct, 'tied_full', 3)

    GMM_llr_fct = np.hstack(GMM_llr_fct)
    PCA_m9_mvg = np.hstack(PCA_m9_mvg)
    Roc_curve_compare(PCA_m9_mvg, GMM_llr_fct, labels, "MVG", "GMM", folder='validation/')
    Bayes_error_plot_compare(PCA_m9_mvg, GMM_llr_fct, labels, "MVG", "GMM", folder='validation/')


# Dte è il fold selezionato, Dtr è tutto il resto
def compute_MVG_score(Dtr, Ltr, Dte, MVG_res):
    llrs_MVG = MVG(Dtr, Ltr, Dte)

    MVG_res.append(llrs_MVG)
    return MVG_res

def ll_GMM(Dtr, Ltr, Dte, llr, typeOf, comp):
    optimal_alpha = 0.1
    llr.extend(GMM_Full(Dtr, Dte, Ltr, optimal_alpha, 2 ** comp, typeOf).tolist())
    return llr