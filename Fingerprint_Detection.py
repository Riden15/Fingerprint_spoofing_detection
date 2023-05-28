import numpy
import matplotlib
import matplotlib.pyplot as plt

import Generative_models
import PCA_LDA
import Tests

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

    print(Tests.split_db_and_try_models(D, L))

    print("---------------------------------------------------------------")

    Tests.Test_split_with_optimal_number_of_PC(D, L)

    print("---------------------------------------------------------------")

    Tests.Test_kFold_with_optimal_number_of_PC(D, L, D.shape[1])

    P = PCA_LDA.PCA(D, 8)
    D_PCA = (numpy.dot(P.T, D))
    P = PCA_LDA.PCA(Dt, 8)
    Dt_PCA = (numpy.dot(P.T, Dt))
    hyp = Generative_models.Gaussian_classify_prova(D_PCA, L)  # usa tutti i dati di train
    acc = Generative_models.Test(hyp, Dt_PCA, Lt)
    print(acc)



