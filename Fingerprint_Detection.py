import numpy

from Validation.validation_MVG import *
from Validation.validation_LR import *

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

def validation(DTR, LTR):
    print("############    MVG    ##############")
    validation_MVG(DTR,LTR,155)
    # con k = 5 i risultati fanno schifo

    print("###########      LR      ##############")
    #L = [0.0001, 0.00001, 1.0, 0.001]
    #validation_LR(DTR,LTR, L , 'LR ')

if __name__ == '__main__':
    D, L = load('Data/Train.txt')
    Dt, Lt = load('Data/Test.txt')
    validation(D, L)


'''
    print(Tests.split_db_and_try_models(D, L))

    print("---------------------------------------------------------------")

    Tests.Test_split_with_optimal_number_of_PC(D, L)

    print("---------------------------------------------------------------")

    Tests.Test_kFold_with_optimal_number_of_PC(D, L, D.shape[1])

    P = PCA_LDA.PCA(D, 5)
    D_PCA = (numpy.dot(P.T, D))
    P = PCA_LDA.PCA(Dt, 5)
    Dt_PCA = (numpy.dot(P.T, Dt))
    hyp = Generative_models.Gaussian_classify_prova(D_PCA, L)  # usa tutti i dati di train
    acc = Generative_models.Test(hyp, Dt_PCA, Lt)
    print(acc)
'''


