import numpy
import matplotlib
import matplotlib.pyplot as plt

def compute_confusion_matrix(nClasses, pred, real):
    CM = numpy.zeros((nClasses, nClasses), dtype=numpy.intc)
    for i in range(real.shape[0]):
        CM[pred[i]][int(real[i])] += 1
    return CM

def confusion_matrix_binary(pred_label, labelsEval):
    C = numpy.zeros((2, 2))
    C[0, 0] = ((pred_label == 0) * (labelsEval == 0)).sum()
    C[0, 1] = ((pred_label == 0) * (labelsEval == 1)).sum()
    C[1, 0] = ((pred_label == 1) * (labelsEval == 0)).sum()
    C[1, 1] = ((pred_label == 1) * (labelsEval == 1)).sum()
    return C

def compute_CostMatrix(Predictions, C):
    CostMatrix = numpy.zeros((Predictions.shape[0], Predictions.shape[1]))
    for i in range(Predictions.shape[1]):
        for j in range(C.shape[0]):
            for z in range(C.shape[1]):
                CostMatrix[j, i] += numpy.dot(Predictions[z, i], C[j, z])
    return CostMatrix


def compute_optimal_Bayes_decision(C, Predictions, labelsEval):
    CostMatrix = compute_CostMatrix(Predictions, C)

    PredictedLabel = numpy.argmin(CostMatrix, axis=0)
    NCorrect = (PredictedLabel.ravel() == labelsEval.ravel()).sum()
    NTotal = labelsEval.size
    # CONFUSION MATRIX
    CM = compute_confusion_matrix(int(labelsEval.max() + 1), PredictedLabel, labelsEval)
    Accuracy = float(NCorrect) / float(NTotal)

    return Accuracy, CM


def compute_optimal_Bayes_decision_Binary_withT(piT, C, predictions, labelsEval):
    CostMatrix = compute_CostMatrix(predictions, C)

    # C[0,1] = Costo dei falsi negativi
    # C[1,0] = Costo dei falsi positivi
    t = numpy.log((piT * C[0, 1]) / ((1 - piT) * C[1, 0])) * -1  # compute the threshold

    # log(Cfn * P(C=1) | x) / log(Cfn * P(C=0) | x) non so perchè è invertito TODO
    final = numpy.log(CostMatrix[0] / CostMatrix[1])

    pred_label = []
    for i in range(final.size):
        if final[i] > t:
            pred_label.append(1)
        else:
            pred_label.append(0)

    acc = sum(pred_label == labelsEval) / labelsEval.shape[0]
    conf_matrix = compute_confusion_matrix(int(labelsEval.max()) + 1, pred_label, labelsEval)
    return acc, conf_matrix


def compute_bayes_risk_DCFu_Binary(piT, C, confuse_matrix):
    FNR = confuse_matrix[0][1] / (confuse_matrix[0][1] + confuse_matrix[1][1])  # false negative rate
    # FNR è la prima riga della colonna di destra (quelli classificati come
    # 0 ma appartenenti alla classe 1)diviso la somma dell'intera colonna
    FPR = confuse_matrix[1][0] / (confuse_matrix[1][0] + confuse_matrix[0][0])  # false positive rate
    # FPR è la stessa cosa ma dell'altra colonna

    # C[0,1] = Costo dei falsi negativi
    # C[1,0] = Costo dei falsi positivi
    dfcu = piT * C[0, 1] * FNR + (1 - piT) * C[1, 0] * FPR
    return dfcu

def compute_bayes_risk_DCF_Binary(piT, C, confuse_matrix):
    dfcu = compute_bayes_risk_DCFu_Binary(piT, C, confuse_matrix)
    min = numpy.min((piT * C[0, 1], (1 - piT) * C[1, 0]))
    dfc = dfcu / min  # dfc è dfcu diviso il minimo tra piT*Cfn e (1-piT)*Cfp
    return dfc

def compute_dcf_min(piT, C, llr, labelsEval):
    thresholdList = numpy.array(llr)
    thresholdList.sort()
    dcfList = []
    for threshold in thresholdList:
        pred_label = numpy.int32(llr > threshold)
        conf_matrix = confusion_matrix_binary(pred_label, labelsEval)
        dcfList.append(compute_bayes_risk_DCF_Binary(piT, C, conf_matrix))
    return numpy.array(dcfList).min()

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


def Bayes_error_plot(C, predictions, labelsEval, confuse_matrix):
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