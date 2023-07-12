
import numpy
import matplotlib
import matplotlib.pyplot as plt
from Utility_functions.Validators import *
import seaborn as sns
import scipy.linalg
from Utility_functions.General_functions import *

def Roc_curve_compare(predictions_firstModel, prediction_secondModel, labelsEval, title1, title2, folder=''):
    C = numpy.array([[0, 1], [10, 0]])

    thresholds = numpy.array(predictions_firstModel)
    thresholds.sort()

    rate_matrix_firstModel = numpy.zeros((2, len(thresholds)))
    index = 0
    for t in thresholds:
        pred_label = numpy.int32(predictions_firstModel > t)
        conf_matrix = compute_confusion_matrix(pred_label, labelsEval)
        FPR = conf_matrix[1, 0] / (conf_matrix[1, 0] + conf_matrix[0, 0])
        TPR = conf_matrix[1, 1] / (conf_matrix[0, 1] + conf_matrix[1, 1])
        rate_matrix_firstModel[:, index] = [FPR, TPR]
        index += 1

    rate_matrix_secondModel = numpy.zeros((2, len(thresholds)))
    index = 0
    for t in thresholds:
        pred_label = numpy.int32(prediction_secondModel > t)
        conf_matrix = compute_confusion_matrix(pred_label, labelsEval)
        FPR = conf_matrix[1, 0] / (conf_matrix[1, 0] + conf_matrix[0, 0])
        TPR = conf_matrix[1, 1] / (conf_matrix[0, 1] + conf_matrix[1, 1])
        rate_matrix_secondModel[:, index] = [FPR, TPR]
        index += 1

    plt.figure()
    plt.plot(rate_matrix_firstModel[0], rate_matrix_firstModel[1], label=title1, color='r')
    plt.plot(rate_matrix_secondModel[0], rate_matrix_secondModel[1], label=title2, color='b')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig('images/comparison/' + folder + 'ROC_' + title1 + '&' + title2 + '.png')
    plt.show()


# serve per plottare, vedere le differenze tra il DCF
def Bayes_error_plot_compare(predictions_firstModel, prediction_secondModel, labelsEval, title1, title2, folder=''):
    C = numpy.array([[0, 1], [1, 0]])
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    dcf_array_1 = [0] * effPriorLogOdds.size
    mindcf_array_1 = [0] * effPriorLogOdds.size
    dcf_array_2 = [0] * effPriorLogOdds.size
    mindcf_array_2 = [0] * effPriorLogOdds.size
    for i in range(effPriorLogOdds.size):
        effective_prior = 1 / (1 + numpy.exp(-effPriorLogOdds[i]))
        dcf_array_1[i] = compute_act_DCF(effective_prior, C, predictions_firstModel, labelsEval)
        mindcf_array_1[i] = compute_dcf_min(effective_prior, predictions_firstModel, labelsEval)
        dcf_array_2[i] = compute_act_DCF(effective_prior, C, prediction_secondModel, labelsEval)
        mindcf_array_2[i] = compute_dcf_min(effective_prior, prediction_secondModel, labelsEval)

    plt.figure()
    plt.plot(effPriorLogOdds, dcf_array_1, label='DCF ' + title1, color='r', linestyle='dashed')
    plt.plot(effPriorLogOdds, mindcf_array_1, label='min DCF ' + title1, color='r')
    plt.plot(effPriorLogOdds, dcf_array_2, label='DCF ' + title2, color='b', linestyle='dashed')
    plt.plot(effPriorLogOdds, mindcf_array_2, label='min DCF ' + title2, color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('DCF')
    plt.legend()
    plt.savefig('images/comparison/' + folder + 'DCF_' + title1 + '&' + title2 + '.png')
    plt.show()

def plot_DCF(x, y, xlabel, title, base=10, folder=''):
    plt.figure()
    plt.plot(x, y[0], label= 'min DCF prior=0.9', color='g')
    plt.plot(x, y[1], label= 'min DCF prior=0.5', color='b')
    plt.plot(x, y[2], label= 'min DCF prior=0.1', color='r')
    plt.xlim([min(x), max(x)])
    plt.xscale("log", base=base)
    plt.legend([ "min DCF prior=0.9", "min DCF prior=0.5", "min DCF prior=0.1"])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.savefig('images/' + folder + title+ '.png')
    plt.show()
    return

def plot_DCF_PCA(x, y, xlabel, title, base=10, folder=''):
    plt.figure()
    plt.plot(x, y[0], label= 'min DCF prior=0.9', color='g')
    plt.plot(x, y[1], label= 'min DCF prior=0.5', color='b')
    plt.plot(x, y[2], label= 'min DCF prior=0.1', color='r')
    plt.plot(x, y[3], label='min DCF prior=0.9 (PCA=9)', color='g', linestyle='dashed')
    plt.plot(x, y[4], label='min DCF prior=0.5 (PCA=9)', color='b', linestyle='dashed')
    plt.plot(x, y[5], label='min DCF prior=0.1 (PCA=9)', color='r', linestyle='dashed')
    plt.xlim([min(x), max(x)])
    plt.xscale("log", base=base)
    plt.legend([ "min DCF prior=0.9", "min DCF prior=0.5", "min DCF prior=0.1", "min DCF prior=0.9 (PCA=9)", "min DCF prior=0.5 (PCA=9)", "min DCF prior=0.1 (PCA=9)"])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.savefig('images/' + folder + title + '.png')
    plt.show()
    return

def plot_DCF_for_SVM_RBF_calibration(x, y, xlabel, title, base=10, folder=''):
    plt.figure()
    plt.plot(x, y[0], label= 'logγ=-4', color='b')
    plt.plot(x, y[1], label= 'logγ=-3', color='g')
    plt.plot(x, y[2], label= 'logγ=-2', color='r')
    plt.plot(x, y[3], label= 'logγ=-1', color='y')
    plt.xlim([min(x), max(x)])
    plt.ylim([0.0, 1.0])
    plt.xscale("log", base=base)
    plt.legend([ "logγ=-4", "logγ=-3", "logγ=-2", "logγ=-1"])
    plt.xlabel(xlabel)
    plt.ylabel("minDCF")
    plt.savefig('images/' + folder + title+ '.png')
    plt.show()
    return


def plot_minDCF_GMM(score_raw, title, components, folder=''):
    labels = numpy.exp2(components).astype(int)

    x = numpy.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    plt.bar(x, score_raw, width, label='minDCF (pi=0.5) - RAW')
    plt.xticks(x, labels)
    plt.ylabel("DCF")
    plt.title(title)
    plt.legend()
    plt.savefig('./images/' + folder + title + 'component_comparison.png')
    plt.show()

