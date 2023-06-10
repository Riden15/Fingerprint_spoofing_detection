import numpy as np

from Utility_functions.Validators import *
from Utility_functions.plot_validators import *
from Models.Logistic_Regression import *
from prettytable import PrettyTable
from PCA_LDA import *

def kfold_QUAD_LR(DTR, LTR, l, appendToTitle, k):
    FoldedData_List = numpy.split(DTR, k, axis=1)
    FoldedLabel_List = numpy.split(LTR, k)

    scores_append = []
    PCA_LR_scores_append = []
    PCA2_LR_scores_append = []
    LR_labels = []

    for i in range(k):
        D = []
        L = []
        if i == 0:
            D.append(np.hstack(FoldedData_List[i + 1:]))
            L.append(np.hstack(FoldedLabel_List[i + 1:]))
        elif i == k - 1:
            D.append(np.hstack(FoldedData_List[:i]))
            L.append(np.hstack(FoldedLabel_List[:i]))
        else:
            D.append(np.hstack(FoldedData_List[:i]))
            D.append(np.hstack(FoldedData_List[i + 1:]))
            L.append(np.hstack(FoldedLabel_List[:i]))
            L.append(np.hstack(FoldedLabel_List[i + 1:]))

        def vecxxT(x):
            x = x[:, None]
            xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
            return xxT

        D = np.hstack(D)
        L = np.hstack(L)

        Dte = FoldedData_List[i]
        Lte = FoldedLabel_List[i]

        expanded_DTR = numpy.apply_along_axis(vecxxT, 0, D)
        expanded_DTE = numpy.apply_along_axis(vecxxT, 0, Dte)
        phi = numpy.vstack([expanded_DTR, D])

        phi_DTE = numpy.vstack([expanded_DTE, Dte])

        scores = quad_logistic_reg_score(phi, L, phi_DTE, l, pi)
        scores_append.append(scores)

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)

        if PCA_Flag is True:
            # PCA m=10
            P = PCA(D, L, m=10)
            DTR_PCA = numpy.dot(P.T, D)
            DTE_PCA = numpy.dot(P.T, Dte)

            PCA_LR_scores = quad_logistic_reg_score(DTR_PCA, L, DTE_PCA, l, pi)
            PCA_LR_scores_append.append(PCA_LR_scores)

            # PCA m=9
            P = PCA(D, L, m=9)
            DTR_PCA = numpy.dot(P.T, D)
            DTE_PCA = numpy.dot(P.T, Dte)

            PCA2_LR_scores = quad_logistic_reg_score(DTR_PCA, L, DTE_PCA, l)
            PCA2_LR_scores_append.append(PCA2_LR_scores)

    validate_LR(scores_append, LR_labels, appendToTitle, l, pi)

    if PCA_Flag is True:
        validate_LR(PCA_LR_scores_append, LR_labels, appendToTitle + 'PCA_m10_', l, pi)

        validate_LR(PCA2_LR_scores_append, LR_labels, appendToTitle + 'PCA_m9_', l, pi)


def validation_quad_LR(DTR, LTR, L, appendToTitle, k):
    for l in L:
        kfold_QUAD_LR(DTR, LTR, l, appendToTitle)


    x = numpy.logspace(-5, 1, 20)
    y = numpy.array([])
    y_05 = numpy.array([])
    y_09 = numpy.array([])
    y_01 = numpy.array([])
    for xi in x:
        scores, labels = kfold_QUAD_LR_tuning(DTR, LTR, xi, PCA_Flag, gauss_Flag, zscore_Flag)
        y_05 = numpy.hstack((y_05, bayes_error_plot_compare(0.5, scores, labels)))
        y_09 = numpy.hstack((y_09, bayes_error_plot_compare(0.9, scores, labels)))
        y_01 = numpy.hstack((y_01, bayes_error_plot_compare(0.1, scores, labels)))

    y = numpy.hstack((y, y_05))
    y = numpy.vstack((y, y_09))
    y = numpy.vstack((y, y_01))

    plot_DCF(x, y, 'lambda', appendToTitle + 'QUAD_LR_minDCF_comparison')
