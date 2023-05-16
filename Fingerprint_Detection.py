import numpy
import matplotlib
import matplotlib.pyplot as plt

import Generative_models
import PCA_LDA

def mcol(v):
    return v.reshape((v.size, 1))

def vrow(vec):
    return vec.reshape((1,vec.shape[0]))

def load(fname):
    DList = []
    # label=0 -> spoofer fingerprint
    # label=1 -> fingerprint
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:10]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = line.split(',')[-1].strip()
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass            
    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)

if __name__ == '__main__':
    D, L = load('Data/Train.txt')

    (DTR, LTR), (DTE, LTE) = Generative_models.split_db_2to1(D, L)
    acc = Generative_models.Gaussian_classify(DTR, LTR, DTE, LTE)
    print(acc)
    acc = Generative_models.Gaussian_classify_log(DTR, LTR, DTE, LTE)
    print(acc)
    acc = Generative_models.Naive_Bayes_Gaussian_classify(DTR, LTR, DTE, LTE)
    print(acc)
    acc = Generative_models.Tied_Covariance_Gaussian_classifier(DTR, LTR, DTE, LTE)
    print(acc)
    acc = Generative_models.Tied_Naive_Covariance_Gaussian_classifier(DTR, LTR, DTE, LTE)
    print(acc)

    print("------")

    P = PCA_LDA.PCA(D, 5)
    D1 = (numpy.dot(P.T, D))
    (DTR, LTR), (DTE, LTE) = Generative_models.split_db_2to1(D1, L)
    acc = Generative_models.Gaussian_classify(DTR, LTR, DTE, LTE)
    print(acc)
    acc = Generative_models.Gaussian_classify_log(DTR, LTR, DTE, LTE)
    print(acc)
    acc = Generative_models.Naive_Bayes_Gaussian_classify(DTR, LTR, DTE, LTE)
    print(acc)
    acc = Generative_models.Tied_Covariance_Gaussian_classifier(DTR, LTR, DTE, LTE)
    print(acc)
    acc = Generative_models.Tied_Naive_Covariance_Gaussian_classifier(DTR, LTR, DTE, LTE)
    print(acc)

    print("-----")

    P2 = PCA_LDA.LDA1(D1, L, 5)
    D2 = (numpy.dot(P2.T, D1))
    (DTR, LTR), (DTE, LTE) = Generative_models.split_db_2to1(D2, L)
    acc = Generative_models.Gaussian_classify(DTR, LTR, DTE, LTE)
    print(acc)
    acc = Generative_models.Gaussian_classify_log(DTR, LTR, DTE, LTE)
    print(acc)
    acc = Generative_models.Naive_Bayes_Gaussian_classify(DTR, LTR, DTE, LTE)
    print(acc)
    acc = Generative_models.Tied_Covariance_Gaussian_classifier(DTR, LTR, DTE, LTE)
    print(acc)
    acc = Generative_models.Tied_Naive_Covariance_Gaussian_classifier(DTR, LTR, DTE, LTE)
    print(acc)

    print("-----")

    Generative_models.kFold_AllTests(D, L, D.shape[1])



