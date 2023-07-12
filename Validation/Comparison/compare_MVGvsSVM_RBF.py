import sys
import numpy as np

from Models.SVM import RBF_KernelFunction
from Utility_functions.plot_validators import Roc_curve_compare, Bayes_error_plot_compare

from Models.Generative_models import *
from Models.PCA_LDA import *

def compare_MVG_vs_SVM_RBF(DTR, LTR, k):
    FoldedData_List = np.split(DTR, k, axis=1)  # lista di fold
    FoldedLabel_List = np.split(LTR, k)

    # lista di llr, una lista per ogni k
    labels = []

    PCA_m9_mvg = []
    PCA_m8_SVM = []

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

        ''' SCORE PCA WITH 8 DIMENSIONS SVM RBF '''
        s, P = PCA(Dtr, m=8)
        DTR_PCA = numpy.dot(P.T, Dtr)
        DTE_PCA = numpy.dot(P.T, Dte)
        score = RBF_KernelFunction(DTR_PCA, Ltr, DTE_PCA, 10, 0.1, 0.001)
        PCA_m8_SVM.append(score)

    PCA_m9_mvg = np.hstack(PCA_m9_mvg)
    PCA_m8_SVM = np.hstack(PCA_m8_SVM)
    Roc_curve_compare(PCA_m9_mvg, PCA_m8_SVM, labels, "MVG", "SVM_RBF", folder='validation/')
    Bayes_error_plot_compare(PCA_m9_mvg, PCA_m8_SVM, labels, "MVG", "SVM_RBF", folder='validation/')




# Dte è il fold selezionato, Dtr è tutto il resto
def compute_MVG_score(Dtr, Ltr, Dte, MVG_res):
    llrs_MVG = MVG(Dtr, Ltr, Dte)

    MVG_res.append(llrs_MVG)
    return MVG_res