import numpy
import matplotlib
import matplotlib.pyplot as plt

def compute_confusion_matrix(pred, real):
    CM = numpy.zeros((int(real.max()) + 1, int(real.max()) + 1), dtype=numpy.intc)
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

def compute_effPrior(pi, C):
    return (pi * C[0, 1]) / ((pi * C[0, 1]) + ((1 - pi) * C[1, 0]))

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
    CM = compute_confusion_matrix(PredictedLabel, labelsEval)
    Accuracy = float(NCorrect) / float(NTotal)

    return Accuracy, CM

def compute_optimal_Bayes_decision_Binary_withT(piT, C, predictions, labelsEval):

    '''
    CostMatrix = compute_CostMatrix(predictions, C)

    # C[0,1] = Costo dei falsi negativi
    # C[1,0] = Costo dei falsi positivi
    t = numpy.log((piT * C[0, 1]) / ((1 - piT) * C[1, 0])) * -1  # compute the threshold

    # log(Cfn * P(C=1) | x) / log(Cfn * P(C=0) | x) non so perchè è invertito TODO

    final = numpy.log(CostMatrix[0] / CostMatrix[1])
'''
    t = numpy.log((piT * C[0, 1]) / ((1 - piT) * C[1, 0])) * -1  # compute the threshold
    pred_label = numpy.int32(predictions > t)
    conf_matrix = compute_confusion_matrix(pred_label, labelsEval)
    return conf_matrix


def compute_bayes_risk_DCFu_Binary(piT, C, confuse_matrix):
    FNR = confuse_matrix[0,1] / (confuse_matrix[0,1] + confuse_matrix[1,1])  # false negative rate
    # FNR è la prima riga della colonna di destra (quelli classificati come
    # 0 ma appartenenti alla classe 1)diviso la somma dell'intera colonna
    FPR = confuse_matrix[1,0] / (confuse_matrix[1,0] + confuse_matrix[0,0])  # false positive rate
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

def compute_act_DCF(piT, C, prediction, labelsEval):
    confuse_matrix = compute_optimal_Bayes_decision_Binary_withT(piT, C, prediction, labelsEval)
    dfcu = compute_bayes_risk_DCFu_Binary(piT, C, confuse_matrix)
    min = numpy.min((piT * C[0, 1], (1 - piT) * C[1, 0]))
    dfc = dfcu / min  # dfc è dfcu diviso il minimo tra piT*Cfn e (1-piT)*Cfp
    return dfc

def compute_dcf_min_effPrior(pi, scores, labels):
    C = numpy.array([[0, 1], [10, 0]])
    effPi = compute_effPrior(pi, C)
    mindcf = compute_dcf_min(effPi, scores, labels)
    return mindcf

def compute_dcf_min(effPi, llr, labelsEval):
    C = numpy.array([[0, 1], [1, 0]])
    thresholdList = numpy.array(llr)
    thresholdList.sort()
    dcfList = []
    for threshold in thresholdList:
        pred_label = numpy.int32(llr > threshold)
        conf_matrix = confusion_matrix_binary(pred_label, labelsEval)
        dcfList.append(compute_bayes_risk_DCF_Binary(effPi, C, conf_matrix))
    return numpy.array(dcfList).min()