import sys
import numpy as np

sys.path.append('../')
from Models.Generative_models import *
from Utility_functions.Validators import *
from prettytable import PrettyTable
from Models.PCA_LDA import *

def evaluation_MVG(DTR, LTR, DTE, LTE):

    # lista di llr, una lista per ogni k
    MVG_res = []
    MVG_naive = []
    MVG_tied = []
    MVG_nt = []
    MVG_labels = []

    # PCA con m = 5
    PCA_m9_mvg = []
    PCA_m9_mvg_naive = []
    PCA_m9_mvg_tied = []
    PCA_m9_mvg_nt = []

    # PCA con m = 8
    PCA_m8_mvg = []
    PCA_m8_mvg_naive = []
    PCA_m8_mvg_tied = []
    PCA_m8_mvg_nt = []

    MVG_labels = np.append(MVG_labels, LTE, axis=0)
    MVG_labels = np.hstack(MVG_labels)

    ''' RAW DATA '''
    MVG_res, MVG_naive, MVG_tied, MVG_nt = compute_MVG_score(DTR, LTR, DTE, MVG_res, MVG_naive, MVG_tied, MVG_nt)

    ''' SCORE PCA WITH 9 DIMENSIONS '''
    s, P = PCA(DTR, m=9)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    PCA_m9_mvg, PCA_m9_mvg_naive, PCA_m9_mvg_tied, PCA_m9_mvg_nt = compute_MVG_score(DTR_PCA, LTR, DTE_PCA,
                                                                                         PCA_m9_mvg, PCA_m9_mvg_naive,
                                                                                         PCA_m9_mvg_tied, PCA_m9_mvg_nt)

    ''' SCORE PCA WITH 8 DIMENSIONS '''
    s, P = PCA(DTR, m=8)
    DTR_PCA = numpy.dot(P.T, DTR)
    DTE_PCA = numpy.dot(P.T, DTE)
    PCA_m8_mvg, PCA_m8_mvg_naive, PCA_m8_mvg_tied, PCA_m8_mvg_nt = compute_MVG_score(DTR_PCA, LTR, DTE_PCA,
                                                                                         PCA_m8_mvg, PCA_m8_mvg_naive,
                                                                                         PCA_m8_mvg_tied, PCA_m8_mvg_nt)

    # π = 0.5 (our application prior), RAW DATA
    evaluation("MVG, RAW data, π=0.5", 0.5, MVG_res, MVG_naive, MVG_tied, MVG_nt, MVG_labels)
    # π = 0.1
    evaluation("MVG, RAW data, π=0.1", 0.1, MVG_res, MVG_naive, MVG_tied, MVG_nt, MVG_labels)
    # π = 0.9
    evaluation("MVG, RAW data, π=0.9", 0.9, MVG_res, MVG_naive, MVG_tied, MVG_nt, MVG_labels)


    # π = 0.5 (our application prior), PCA m=9
    evaluation("MVG, PCA m=9, π=0.5", 0.5, PCA_m9_mvg, PCA_m9_mvg_naive, PCA_m9_mvg_tied, PCA_m9_mvg_nt, MVG_labels)
    # π = 0.1, PCA m=9
    evaluation("MVG, PCA m=9, π=0.1", 0.1, PCA_m9_mvg, PCA_m9_mvg_naive, PCA_m9_mvg_tied, PCA_m9_mvg_nt, MVG_labels)
    # π = 0.9 , PCA m=9
    evaluation("MVG, PCA m=9, π=0.9", 0.9, PCA_m9_mvg, PCA_m9_mvg_naive, PCA_m9_mvg_tied, PCA_m9_mvg_nt, MVG_labels)

    # π = 0.5 (our application prior), PCA m=8
    evaluation("MVG, PCA m=8, π=0.5", 0.5, PCA_m8_mvg, PCA_m8_mvg_naive, PCA_m8_mvg_tied, PCA_m8_mvg_nt, MVG_labels)
    # π = 0.1 , PCA m=8
    evaluation("MVG, PCA m=8, π=0.1", 0.1, PCA_m8_mvg, PCA_m8_mvg_naive, PCA_m8_mvg_tied, PCA_m8_mvg_nt, MVG_labels)
    # π = 0.9 , PCA m=8
    evaluation("MVG, PCA m=8, π=0.9", 0.9, PCA_m8_mvg, PCA_m8_mvg_naive, PCA_m8_mvg_tied, PCA_m8_mvg_nt, MVG_labels)


# Dte è il fold selezionato, Dtr è tutto il resto
def compute_MVG_score(Dtr, Ltr, Dte, MVG_res, MVG_naive, MVG_t, MVG_nt):
    llrs_MVG = MVG(Dtr, Ltr, Dte)
    llrs_naive = Naive_Bayes_Gaussian_classify(Dtr, Ltr, Dte)
    llrs_tied = Tied_Covariance_Gaussian_classifier(Dtr, Ltr, Dte)
    llrs_nt = Tied_Naive_Covariance_Gaussian_classifier(Dtr, Ltr, Dte)

    MVG_res.append(llrs_MVG)
    MVG_naive.append(llrs_naive)
    MVG_t.append(llrs_tied)
    MVG_nt.append(llrs_nt)
    return MVG_res, MVG_naive, MVG_t, MVG_nt


def evaluation(title, pi, MVG_res, MVG_naive, MVG_tied, MVG_nt, MVG_labels):
    MVG_res = np.hstack(MVG_res)
    MVG_naive = np.hstack(MVG_naive)
    MVG_tied = np.hstack(MVG_tied)
    MVG_nt = np.hstack(MVG_nt)

    llrs_tot = compute_dcf_min_effPrior(pi, MVG_res, MVG_labels)
    llrs_naive_tot = compute_dcf_min_effPrior(pi, MVG_naive, MVG_labels)
    llrs_tied_tot = compute_dcf_min_effPrior(pi, MVG_tied, MVG_labels)
    llrs_nt_tot = compute_dcf_min_effPrior(pi, MVG_nt, MVG_labels)

    t = PrettyTable(["Type", "minDCF"])
    t.title = title
    t.add_row(["MVG", round(llrs_tot, 3)])
    t.add_row(["MVG naive", round(llrs_naive_tot, 3)])
    t.add_row(["MVG tied", round(llrs_tied_tot, 3)])
    t.add_row(["MVG naive + tied", round(llrs_nt_tot, 3)])
    print(t)

    # plot_ROC(MVG_res, MVG_labels, appendToTitle + 'MVG')
    # plot_ROC(MVG_naive, MVG_labels, appendToTitle + 'MVG + Naive')
    # plot_ROC(MVG_t, MVG_labels, appendToTitle + 'MVG + Tied')
    # plot_ROC(MVG_nt, MVG_labels, appendToTitle + 'MVG + Naive + Tied')

    # # Cfn and Ctp are set to 1
    # bayes_error_min_act_plot(MVG_res, MVG_labels, appendToTitle + 'MVG', 0.4)
    # bayes_error_min_act_plot(MVG_naive, MVG_labels, appendToTitle + 'MVG + Naive', 1)
    # bayes_error_min_act_plot(MVG_t, MVG_labels, appendToTitle + 'MVG + Tied', 0.4)
    # bayes_error_min_act_plot(MVG_nt, MVG_labels, appendToTitle + 'MVG + Naive + Tied', 1)