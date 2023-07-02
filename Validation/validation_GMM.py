# -*- coding: utf-8 -*-
import sys

import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
from prettytable import PrettyTable

sys.path.append('../')
from Models.GMM import *
from Utility_functions.Validators import compute_dcf_min_effPrior
from Utility_functions.plot_validators import plot_minDCF_GMM
from PCA_LDA import *


def validation_GMM_tot(DTR, LTR, k):
    # We'll train from 1 to 2^7 components
    componentsToTry = [1, 2, 3, 4, 5, 6, 7]
    for comp in componentsToTry:
        kfold_GMM(DTR, LTR, comp, k)

'''
    bars_full_cov = numpy.array([])
    bars_diag_cov = numpy.array([])
    bars_tied_full_cov = numpy.array([])
    bars_tied_diag_cov = numpy.array([])
    for comp in componentsToTry:
        GMM_llr_fc, GMM_llr_dc, GMM_llr_fct, GMM_llr_dct, labels = kfold_GMM_calibration(DTR, LTR, comp, k)
        bars_full_cov = numpy.hstack((bars_full_cov, compute_dcf_min_effPrior(0.5, GMM_llr_fc, labels)))
        bars_diag_cov = numpy.hstack((bars_diag_cov, compute_dcf_min_effPrior(0.5, GMM_llr_dc, labels)))
        bars_tied_full_cov = numpy.hstack((bars_tied_full_cov, compute_dcf_min_effPrior(0.5, GMM_llr_fct, labels)))
        bars_tied_diag_cov = numpy.hstack((bars_tied_diag_cov, compute_dcf_min_effPrior(0.5, GMM_llr_dct, labels)))

    plot_minDCF_GMM(bars_full_cov, 'full-cov', componentsToTry)
    plot_minDCF_GMM(bars_diag_cov, 'diag-cov', componentsToTry)
    plot_minDCF_GMM(bars_tied_full_cov, 'tied_full-cov', componentsToTry)
    plot_minDCF_GMM(bars_tied_diag_cov, 'tied_diag-cov', componentsToTry)
'''


def kfold_GMM(DTR, LTR, comp, k):
    FoldedData_List = np.split(DTR, k, axis=1)
    FoldedLabel_List = np.split(LTR, k)

    GMM_llr_fc = []
    GMM_llr_dc = []
    GMM_llr_fct = []
    GMM_llr_dct = []

    GMM_llr_fc_PCA = []
    GMM_llr_dc_PCA = []
    GMM_llr_fct_PCA = []
    GMM_llr_dct_PCA = []

    GMM_labels = []

    for fold in range(k):
        Dtr, Ltr, Dte, Lte = kfold(fold, k, FoldedData_List, FoldedLabel_List)
        GMM_labels = np.append(GMM_labels, Lte)
        GMM_labels = np.hstack(GMM_labels)

        # RAW DATA
        # full-cov
        GMM_llr_fc = ll_GMM(Dtr, Ltr, Dte, GMM_llr_fc, 'full', comp)
        # diag-cov
        GMM_llr_dc = ll_GMM(Dtr, Ltr, Dte, GMM_llr_dc, 'diag', comp)
        # full-cov tied
        GMM_llr_fct = ll_GMM(Dtr, Ltr, Dte, GMM_llr_fct, 'tied_full', comp)
        # diag-cov tied
        GMM_llr_dct = ll_GMM(Dtr, Ltr, Dte, GMM_llr_dct, 'tied_diag', comp)

        #PCA m=9
        s, P = PCA(Dtr, m=9)
        DTR_PCA = numpy.dot(P.T, Dtr)
        DTE_PCA = numpy.dot(P.T, Dte)
        # full-cov
        GMM_llr_fc_PCA = ll_GMM(Dtr, Ltr, Dte, GMM_llr_fc_PCA, 'full', comp)
        # diag-cov
        GMM_llr_dc_PCA = ll_GMM(Dtr, Ltr, Dte, GMM_llr_dc_PCA, 'diag', comp)
        # full-cov tied
        GMM_llr_fct_PCA = ll_GMM(Dtr, Ltr, Dte, GMM_llr_fct_PCA, 'tied_full', comp)
        # diag-cov tied
        GMM_llr_dct_PCA = ll_GMM(Dtr, Ltr, Dte, GMM_llr_dct_PCA, 'tied_diag', comp)

    '''RAW data pi=0.1'''
    validation_GMM("GMM, RAW data, π=0.1", 0.1, GMM_llr_fc, GMM_llr_dc, GMM_llr_fct, GMM_llr_dct, GMM_labels)
    '''RAW data pi=0.5'''
    validation_GMM("GMM, RAW data, π=0.5", 0.5, GMM_llr_fc, GMM_llr_dc, GMM_llr_fct, GMM_llr_dct, GMM_labels)
    '''RAW data pi=0.9'''
    validation_GMM("GMM, RAW data, π=0.9", 0.9, GMM_llr_fc, GMM_llr_dc, GMM_llr_fct, GMM_llr_dct, GMM_labels)

    '''PCA with m = 9, pi=0.1'''
    validation_GMM("GMM, PCA m=9, π=0.1", 0.1, GMM_llr_fc_PCA, GMM_llr_dc_PCA, GMM_llr_fct_PCA, GMM_llr_dct_PCA, GMM_labels)
    '''PCA with m = 9, pi=0.5'''
    validation_GMM("GMM, PCA m=9, π=0.5", 0.5, GMM_llr_fc_PCA, GMM_llr_dc_PCA, GMM_llr_fct_PCA, GMM_llr_dct_PCA, GMM_labels)
    '''PCA with m = 9, pi=0.9'''
    validation_GMM("GMM, PCA m=9, π=0.9", 0.9, GMM_llr_fc_PCA, GMM_llr_dc_PCA, GMM_llr_fct_PCA, GMM_llr_dct_PCA, GMM_labels)


def validation_GMM(title, pi, GMM_llr_fc, GMM_llr_dc, GMM_llr_fct, GMM_llr_dct, GMM_Labels):
    GMM_llr_fc = np.hstack(GMM_llr_fc)
    GMM_llr_dc = np.hstack(GMM_llr_dc)
    GMM_llr_fct = np.hstack(GMM_llr_fct)
    GMM_llr_dct = np.hstack(GMM_llr_dct)

    llrs_fc_tot = compute_dcf_min_effPrior(pi, GMM_llr_fc, GMM_Labels)
    llrs_dc_tot = compute_dcf_min_effPrior(pi, GMM_llr_dc, GMM_Labels)
    llrs_fct_tot = compute_dcf_min_effPrior(pi, GMM_llr_fct, GMM_Labels)
    llrs_dct_tot = compute_dcf_min_effPrior(pi, GMM_llr_dct, GMM_Labels)

    t = PrettyTable(["Type", "minDCF"])
    t.title = title
    t.add_row(["GMM Full-cov", round(llrs_fc_tot, 3)])
    t.add_row(["GMM Diag-cov", round(llrs_dc_tot, 3)])
    t.add_row(["GMM Full-cov + tied", round(llrs_fct_tot, 3)])
    t.add_row(["GMM Diag-cov + tied", round(llrs_dct_tot, 3)])
    print(t)
    
def ll_GMM(Dtr, Ltr, Dte, llr, typeOf, comp):

    optimal_alpha = 0.1
    llr.extend(GMM_Full(Dtr, Dte, Ltr, optimal_alpha, 2 ** comp, typeOf).tolist())
    return llr

def kfold_GMM_calibration(DTR, LTR, comp, k):
    FoldedData_List = np.split(DTR, k, axis=1)
    FoldedLabel_List = np.split(LTR, k)

    GMM_llr_fc = []
    GMM_llr_dc = []
    GMM_llr_fct = []
    GMM_llr_dct = []
    GMM_labels = []

    for fold in range(k):
        Dtr, Ltr, Dte, Lte = kfold(fold, k, FoldedData_List, FoldedLabel_List)

        GMM_labels = np.append(GMM_labels, Lte)
        GMM_labels = np.hstack(GMM_labels)

        # RAW DATA
        # full-cov
        GMM_llr_fc = ll_GMM(Dtr, Ltr, Dte, GMM_llr_fc, 'full', comp)
        # diag-cov
        GMM_llr_dc = ll_GMM(Dtr, Ltr, Dte, GMM_llr_dc, 'diag', comp)
        # full-cov tied
        GMM_llr_fct = ll_GMM(Dtr, Ltr, Dte, GMM_llr_fct, 'tied_full', comp)
        # diag-cov tied
        GMM_llr_dct = ll_GMM(Dtr, Ltr, Dte, GMM_llr_dct, 'tied_diag', comp)

    return np.hstack(GMM_llr_fc), np.hstack(GMM_llr_dc), np.hstack(GMM_llr_fct), np.hstack(GMM_llr_dct), GMM_labels
