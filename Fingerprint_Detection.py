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
    Dt, Lt = load('Data/Test.txt')

    hyp = Generative_models.Gaussian_classify_prova(D, L)  # usa tutti i dati di train
    acc = Generative_models.Test(hyp, Dt, Lt)
    print(acc)

    # sta roba cra 10 modelli, uno con un hyperparameter del PCA diverso per capire quale è il migliore usando l'evaluation test
    # evaluation test = ultimo 1/3 del train set
    #TODO provare con la kfold anzichè con lo spit --> con essa si usano tutti i sample per capire l'hyperparameter
    for i in range(10):
        P = PCA_LDA.PCA(D, i)
        D_PCA = (numpy.dot(P.T, D))
        (DTR, LTR), (DTE, LTE) = Generative_models.split_db_2to1(D_PCA, L)
        hyp = Generative_models.Gaussian_classify_prova(DTR, LTR)
        acc = Generative_models.Test(hyp, DTE, LTE)
        print(acc, i)

    #Generative_models.split_db_2to1_AllTests(D, L)
    #Generative_models.kFold_AllTests(D, L, D.shape[1])



