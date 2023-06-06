
import numpy
import matplotlib
import matplotlib.pyplot as plt
from Utility_functions.Validators import *

def Roc_curve(C, predictions, labelsEval):
    CostMatrix = compute_CostMatrix(predictions, C)

    # log(Cfn * P(C=1) | x) / log(Cfn * P(C=0) | x) non so perchè è invertito TODO
    final = numpy.log(CostMatrix[0] / CostMatrix[1])

    rate_matrix = numpy.zeros((2, len(final)))
    index = 0
    for t in final:
        pred_label = []
        for i in range(final.size):
            if final[i] > t:
                pred_label.append(1)
            else:
                pred_label.append(0)
        conf_matrix = compute_confusion_matrix(int(labelsEval.max()) + 1, pred_label, labelsEval)
        FPR = conf_matrix[1, 0] / (conf_matrix[1, 0] + conf_matrix[0, 0])
        TPR = conf_matrix[1, 1] / (conf_matrix[0, 1] + conf_matrix[1, 1])
        rate_matrix[:, index] = [FPR, TPR]
        index += 1

    plt.figure()
    plt.scatter(rate_matrix[0], rate_matrix[1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()

def Bayes_error_plot(pi,scores,labels):
    #pi = 1.0 / (1.0 + numpy.exp(-pi))
    C = numpy.array([[0, 1], [10, 0]])
    mindcf = compute_dcf_min(pi, C, scores, labels)
    return mindcf

# serve per plottare, vedere le differenze tra il DCF
def Bayes_error_plot2(C, predictions, labelsEval, confuse_matrix):
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    dcf_array = [0] * effPriorLogOdds.size
    mindcf_array = [0] * effPriorLogOdds.size
    for i in range(effPriorLogOdds.size):
        effective_prior = 1 / (1 + numpy.exp(-effPriorLogOdds[i]))
        dcf_array[i] = compute_bayes_risk_DCF_Binary(effective_prior, C, confuse_matrix)
        mindcf_array[i] = compute_dcf_min(effective_prior, C, predictions, labelsEval)

    plt.figure()
    plt.plot(effPriorLogOdds, dcf_array, label='DCF', color='r')
    plt.plot(effPriorLogOdds, mindcf_array, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('DCF')
    plt.legend()
    plt.show()

def plot_DCF(x, y, xlabel, title, base=10):
    plt.figure()
    plt.plot(x, y[0], label= 'min DCF prior=0.5', color='b')
    plt.plot(x, y[1], label= 'min DCF prior=0.9', color='g')
    plt.plot(x, y[2], label= 'min DCF prior=0.1', color='r')
    plt.xlim([min(x), max(x)])
    plt.xscale("log", base=base)
    plt.legend([ "min DCF prior=0.5", "min DCF prior=0.9", "min DCF prior=0.1"])
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    plt.savefig('images/DCF_' + title+ '.svg')
    plt.show()
    return