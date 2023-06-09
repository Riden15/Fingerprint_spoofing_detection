import sys
import numpy as np

sys.path.append('../')
from Models.Generative_models import *
from Utility_functions.plot_functions import *
from Utility_functions.Validators import *
from prettytable import PrettyTable
from PCA_LDA import *

def validation_MVG(DTR, LTR, k, appendToTitle):
    FoldedData_List = np.split(DTR, k, axis=1) #lista di fold
    FoldedLabel_List = np.split(LTR, k)

    # lista di llr, una lista per ogni k
    MVG_res = []
    MVG_naive = []
    MVG_tied = []
    MVG_nt = []
    MVG_labels = []

    # PCA con m = 5
    PCA_m5_mvg = []
    PCA_m5_mvg_naive = []
    PCA_m5_mvg_tied = []
    PCA_m5_mvg_nt = []

    # PCA, LDA con m=5
    PCA_LDA_m5_mvg = []
    PCA_LDA_m5_mvg_naive = []
    PCA_LDA_m5_mvg_tied = []
    PCA_LDA_m5_mvg_nt = []

    # PCA con m = 8
    PCA_m8_mvg = []
    PCA_m8_mvg_naive = []
    PCA_m8_mvg_tied = []
    PCA_m8_mvg_nt = []

    # PCA, LDA con m=8
    PCA_LDA_m8_mvg = []
    PCA_LDA_m8_mvg_naive = []
    PCA_LDA_m8_mvg_tied = []
    PCA_LDA_m8_mvg_nt = []

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
            Dtr.append(np.hstack(FoldedData_List[:i])) #append da 0 a i-1 poi da i+1 fino alla fine, poi i lo uso come DTE
            Dtr.append(np.hstack(FoldedData_List[i + 1:]))
            Ltr.append(np.hstack(FoldedLabel_List[:i]))
            Ltr.append(np.hstack(FoldedLabel_List[i + 1:]))

        Dtr = np.hstack(Dtr) # fold selezionati per training (dati)
        Ltr = np.hstack(Ltr) # fold selezionati per training (label)

        Dte = FoldedData_List[i]  # singolo fold selezionato per evaluation (dati)
        Lte = FoldedLabel_List[i] # singolo fold selezionato per evaluation (label)

        MVG_labels = np.append(MVG_labels, Lte, axis=0)
        MVG_labels = np.hstack(MVG_labels)
        # Once we have computed our folds, we can try different models

        ''' RAW DATA '''
        MVG_res, MVG_naive, MVG_tied, MVG_nt = compute_MVG_score(Dtr, Ltr, Dte, MVG_res, MVG_naive, MVG_tied, MVG_nt)

        ''' SCORE PCA WITH 5 DIMENSIONS '''
        s, P = PCA(Dtr, m=5)
        DTR_PCA = numpy.dot(P.T, Dtr)
        DTE_PCA = numpy.dot(P.T, Dte)
        PCA_m5_mvg, PCA_m5_mvg_naive, PCA_m5_mvg_tied, PCA_m5_mvg_nt = compute_MVG_score(DTR_PCA, Ltr, DTE_PCA,
                                                                                         PCA_m5_mvg, PCA_m5_mvg_naive,
                                                                                         PCA_m5_mvg_tied, PCA_m5_mvg_nt)

        ''' SCORE PCA_LDA WITH 5 DIMENSIONS'''
        P = PCA_LDA.LDA1(DTR_PCA, Ltr, 5)
        DTR_PCA_LDA = numpy.dot(P.T, DTR_PCA)
        DTE_PCA_LDA = numpy.dot(P.T, DTE_PCA)
        PCA_LDA_m5_mvg, PCA_LDA_m5_mvg_naive, PCA_LDA_m5_mvg_tied, PCA_LDA_m5_mvg_nt = compute_MVG_score(DTR_PCA_LDA, Ltr,
                                                                                                         DTE_PCA_LDA,
                                                                                                         PCA_LDA_m5_mvg,
                                                                                                         PCA_LDA_m5_mvg_naive,
                                                                                                         PCA_LDA_m5_mvg_tied,
                                                                                                         PCA_LDA_m5_mvg_nt)

        ''' SCORE PCA WITH 8 DIMENSIONS '''
        s, P = PCA(Dtr, m=8)
        DTR_PCA = numpy.dot(P.T, Dtr)
        DTE_PCA = numpy.dot(P.T, Dte)
        PCA_m8_mvg, PCA_m8_mvg_naive, PCA_m8_mvg_tied, PCA_m8_mvg_nt = compute_MVG_score(DTR_PCA, Ltr, DTE_PCA,
                                                                                         PCA_m8_mvg, PCA_m8_mvg_naive,
                                                                                         PCA_m8_mvg_tied, PCA_m8_mvg_nt)

        ''' SCORE PCA_LDA WITH 8 DIMENSIONS'''
        P = PCA_LDA.LDA1(DTR_PCA, Ltr, 8)
        DTR_PCA_LDA = numpy.dot(P.T, DTR_PCA)
        DTE_PCA_LDA = numpy.dot(P.T, DTE_PCA)
        PCA_LDA_m8_mvg, PCA_LDA_m8_mvg_naive, PCA_LDA_m8_mvg_tied, PCA_LDA_m8_mvg_nt = compute_MVG_score(DTR_PCA_LDA, Ltr,
                                                                                                         DTE_PCA_LDA,
                                                                                                         PCA_LDA_m8_mvg,
                                                                                                         PCA_LDA_m8_mvg_naive,
                                                                                                         PCA_LDA_m8_mvg_tied,
                                                                                                         PCA_LDA_m8_mvg_nt)

    # π = 0.5 (our application prior), RAW DATA
    evaluation(appendToTitle + "minDCF: π=0.5", 0.5, MVG_res, MVG_naive, MVG_tied, MVG_nt, MVG_labels)
    # π = 0.1
    evaluation(appendToTitle + "minDCF: π=0.1", 0.1, MVG_res, MVG_naive, MVG_tied, MVG_nt, MVG_labels)
    # π = 0.9
    evaluation(appendToTitle + "minDCF: π=0.9", 0.9, MVG_res, MVG_naive, MVG_tied, MVG_nt, MVG_labels)


    # π = 0.5 (our application prior), PCA m=5
    evaluation(appendToTitle + "minDCF: π=0.5, PCA m=5", 0.5, PCA_m5_mvg, PCA_m5_mvg_naive, PCA_m5_mvg_tied, PCA_m5_mvg_nt, MVG_labels)
    # π = 0.1, PCA m=5
    evaluation(appendToTitle + "minDCF: π=0.1, PCA m=5", 0.1, PCA_m5_mvg, PCA_m5_mvg_naive, PCA_m5_mvg_tied, PCA_m5_mvg_nt, MVG_labels)
    # π = 0.9 , PCA m=5
    evaluation(appendToTitle + "minDCF: π=0.9, PCA m=5", 0.9, PCA_m5_mvg, PCA_m5_mvg_naive, PCA_m5_mvg_tied, PCA_m5_mvg_nt, MVG_labels)
    # π = 0.5 (our application prior), PCA, LDA m=5
    evaluation(appendToTitle + "minDCF: π=0.5, PCA, LDA m=5", 0.5, PCA_LDA_m5_mvg, PCA_LDA_m5_mvg_naive, PCA_LDA_m5_mvg_tied, PCA_LDA_m5_mvg_nt, MVG_labels)
    # π = 0.1, PCA, LDA m=5
    evaluation(appendToTitle + "minDCF: π=0.1, PCA, LDA m=5", 0.1, PCA_LDA_m5_mvg, PCA_LDA_m5_mvg_naive, PCA_LDA_m5_mvg_tied, PCA_LDA_m5_mvg_nt, MVG_labels)
    # π = 0.9 , PCA, LDA  m=5
    evaluation(appendToTitle + "minDCF: π=0.9, PCA, LDA m=5", 0.9, PCA_LDA_m5_mvg, PCA_LDA_m5_mvg_naive, PCA_LDA_m5_mvg_tied, PCA_LDA_m5_mvg_nt, MVG_labels)


    # π = 0.5 (our application prior), PCA m=8
    evaluation(appendToTitle + "minDCF: π=0.5, PCA m=8", 0.5, PCA_m8_mvg, PCA_m8_mvg_naive, PCA_m8_mvg_tied, PCA_m8_mvg_nt, MVG_labels)
    # π = 0.1 , PCA m=8
    evaluation(appendToTitle + "minDCF: π=0.1, PCA m=8", 0.1, PCA_m8_mvg, PCA_m8_mvg_naive, PCA_m8_mvg_tied, PCA_m8_mvg_nt, MVG_labels)
    # π = 0.9 , PCA m=8
    evaluation(appendToTitle + "minDCF: π=0.9, PCA m=8", 0.9, PCA_m8_mvg, PCA_m8_mvg_naive, PCA_m8_mvg_tied, PCA_m8_mvg_nt, MVG_labels)
    # π = 0.5 (our application prior), PCA, LDA m=8
    evaluation(appendToTitle + "minDCF: π=0.5, PCA, LDA m=8", 0.5, PCA_LDA_m8_mvg, PCA_LDA_m8_mvg_naive, PCA_LDA_m8_mvg_tied, PCA_LDA_m8_mvg_nt,MVG_labels)
    # π = 0.1, PCA, LDA m=8
    evaluation(appendToTitle + "minDCF: π=0.1, PCA, LDA m=8", 0.1, PCA_LDA_m8_mvg, PCA_LDA_m8_mvg_naive, PCA_LDA_m8_mvg_tied, PCA_LDA_m8_mvg_nt,MVG_labels)
    # π = 0.9 , PCA, LDA  m=8
    evaluation(appendToTitle + "minDCF: π=0.9, PCA, LDA m=8", 0.9, PCA_LDA_m8_mvg, PCA_LDA_m8_mvg_naive, PCA_LDA_m8_mvg_tied, PCA_LDA_m8_mvg_nt,MVG_labels)


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
    # MVG_labels.append(Lte)
    # MVG_labels = np.append(MVG_labels, Lte, axis=0)
    # MVG_labels = np.hstack(MVG_labels)
    return MVG_res, MVG_naive, MVG_t, MVG_nt


def evaluation(title, pi, MVG_res, MVG_naive, MVG_t, MVG_nt, MVG_labels):
    MVG_res = np.hstack(MVG_res)
    MVG_naive = np.hstack(MVG_naive)
    MVG_t = np.hstack(MVG_t)
    MVG_nt = np.hstack(MVG_nt)

    C = np.array([[0, 1], [10, 0]])  # costi Cfp = 10, Cfn = 1

    llrs_tot = compute_dcf_min(pi, C, MVG_res, MVG_labels)
    llrsn_tot = compute_dcf_min(pi, C, MVG_naive, MVG_labels)
    llrst_tot = compute_dcf_min(pi, C, MVG_t, MVG_labels)
    llrsnt_tot = compute_dcf_min(pi, C, MVG_nt, MVG_labels)


    t = PrettyTable(["Type", "minDCF"])
    t.title = title
    t.add_row(["MVG", round(llrs_tot, 3)])
    t.add_row(["MVG naive", round(llrsn_tot, 3)])
    t.add_row(["MVG tied", round(llrst_tot, 3)])
    t.add_row(["MVG naive + tied", round(llrsnt_tot, 3)])
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