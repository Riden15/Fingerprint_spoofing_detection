import numpy as np

from Utility_functions.Validators import *
from Utility_functions.plot_validators import *
from Models.Logistic_Regression import *
from prettytable import PrettyTable
from PCA_LDA import *

def kfold_QUAD_LR(DTR, LTR, l, pi, appendToTitle, k):
    FoldedData_List = numpy.split(DTR, k, axis=1)
    FoldedLabel_List = numpy.split(LTR, k)

    scores_append = []
    PCA_LR_scores_append = []
    PCA2_LR_scores_append = []
    LR_labels = []

    for i in range(k):
        Dtr = []
        Ltr = []
        if i == 0:
            Dtr.append(np.hstack(FoldedData_List[i + 1:]))
            Ltr.append(np.hstack(FoldedLabel_List[i + 1:]))
        elif i == k - 1:
            Dtr.append(np.hstack(FoldedData_List[:i]))
            Ltr.append(np.hstack(FoldedLabel_List[:i]))
        else:
            Dtr.append(np.hstack(FoldedData_List[:i]))
            Dtr.append(np.hstack(FoldedData_List[i + 1:]))
            Ltr.append(np.hstack(FoldedLabel_List[:i]))
            Ltr.append(np.hstack(FoldedLabel_List[i + 1:]))

        def vecxxT(x):
            x = x[:, None]
            xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
            return xxT

        Dtr = np.hstack(Dtr)
        Ltr = np.hstack(Ltr)

        Dte = FoldedData_List[i]
        Lte = FoldedLabel_List[i]

        expanded_DTR = numpy.apply_along_axis(vecxxT, 0, Dtr)
        expanded_DTE = numpy.apply_along_axis(vecxxT, 0, Dte)
        phi = numpy.vstack([expanded_DTR, Dtr])

        phi_DTE = numpy.vstack([expanded_DTE, Dte])

        scores = quad_logistic_reg_score(phi, Ltr, phi_DTE, l, pi)
        scores_append.append(scores)

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)


        # PCA m=9
        s, P = PCA(Dtr, 9)
        DTR_PCA = numpy.dot(P.T, Dtr)
        DTE_PCA = numpy.dot(P.T, Dte)

        PCA_LR_scores = quad_logistic_reg_score(DTR_PCA, Ltr, DTE_PCA, l, pi)
        PCA_LR_scores_append.append(PCA_LR_scores)

        # PCA m=7
        s, P = PCA(Dtr, 7)
        DTR_PCA = numpy.dot(P.T, Dtr)
        DTE_PCA = numpy.dot(P.T, Dte)

        PCA2_LR_scores = quad_logistic_reg_score(DTR_PCA, Ltr, DTE_PCA, l)
        PCA2_LR_scores_append.append(PCA2_LR_scores)

    validate_LR(scores_append, LR_labels, appendToTitle, l, pi)


    validate_LR(PCA_LR_scores_append, LR_labels, appendToTitle + 'PCA_m10_', l, pi)

    validate_LR(PCA2_LR_scores_append, LR_labels, appendToTitle + 'PCA_m9_', l, pi)

def validate_LR(scores, LR_labels, appendToTitle, l, pi):
    scores_append = np.hstack(scores)
    scores_tot_05 = compute_dcf_min_effPrior(0.5, scores_append, LR_labels)
    scores_tot_01 = compute_dcf_min_effPrior(0.1, scores_append, LR_labels)
    scores_tot_09 = compute_dcf_min_effPrior(0.9, scores_append, LR_labels)
    # plot_ROC(scores_append, LR_labels, appendToTitle + 'WEIGHTED_LR, lambda=' + str(l))

    # Cfn and Ctp are set to 1
    # bayes_error_min_act_plot(scores_append, LR_labels, appendToTitle + 'WEIGHTED_LR, lambda=' + str(l), 0.4)

    t = PrettyTable(["Type", "π=0.5", "π=0.1", "π=0.9"])
    t.title = appendToTitle
    t.add_row(['QUAD_LR, lambda=' + str(l) + " π_t=" + str(pi), round(scores_tot_05, 3), round(scores_tot_01, 3), round(scores_tot_09, 3)])
    print(t)


def kfold_QUAD_LR_tuning(DTR, LTR, l):
    k = 5
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)

    scores_append = []
    LR_labels = []

    for i in range(k):
        D = []
        L = []
        if i == 0:
            D.append(np.hstack(Dtr[i + 1:]))
            L.append(np.hstack(Ltr[i + 1:]))
        elif i == k - 1:
            D.append(np.hstack(Dtr[:i]))
            L.append(np.hstack(Ltr[:i]))
        else:
            D.append(np.hstack(Dtr[:i]))
            D.append(np.hstack(Dtr[i + 1:]))
            L.append(np.hstack(Ltr[:i]))
            L.append(np.hstack(Ltr[i + 1:]))

        def vecxxT(x):
            x = x[:, None]
            xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
            return xxT

        D = np.hstack(D)
        L = np.hstack(L)

        Dte = Dtr[i]
        Lte = Ltr[i]

        expanded_DTR = numpy.apply_along_axis(vecxxT, 0, D)
        expanded_DTE = numpy.apply_along_axis(vecxxT, 0, Dte)
        phi = numpy.vstack([expanded_DTR, D])

        phi_DTE = numpy.vstack([expanded_DTE, Dte])

        scores = quad_logistic_reg_score(phi, L, phi_DTE, l)
        scores_append.append(scores)

        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)

    return np.hstack(scores_append), LR_labels


def validation_quad_LR(DTR, LTR, L, appendToTitle, k):
    for l in L:
        kfold_QUAD_LR(DTR, LTR, l, 0.5, appendToTitle, k)
        kfold_QUAD_LR(DTR, LTR, l, 0.1, appendToTitle, k)
        kfold_QUAD_LR(DTR, LTR, l, 0.9, appendToTitle, k)

'''
    x = numpy.logspace(-5, 1, 20)
    y = numpy.array([])
    y_05 = numpy.array([])
    y_09 = numpy.array([])
    y_01 = numpy.array([])
    for xi in x:
        scores, labels = kfold_QUAD_LR_tuning(DTR, LTR, xi)
        y_05 = numpy.hstack((y_05, compute_dcf_min_effPrior(0.5, scores, labels)))
        y_09 = numpy.hstack((y_09, compute_dcf_min_effPrior(0.9, scores, labels)))
        y_01 = numpy.hstack((y_01, compute_dcf_min_effPrior(0.1, scores, labels)))

    y = numpy.hstack((y, y_05))
    y = numpy.vstack((y, y_09))
    y = numpy.vstack((y, y_01))

    plot_DCF(x, y, 'lambda', appendToTitle + 'QUAD_LR_minDCF_comparison')
'''
