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
    egnValues, egnVector = PCA(DTR, 10)
    plot_explained_variance(egnValues)

    print("############    MVG    ##############")
    # validation_MVG(DTR,LTR,155, "MVG, ")
    # con k = 5 i risultati fanno schifo

    print("###########      LR      ##############")
    L = [0.00001]
    #validation_LR(DTR,LTR, L , 'LR, ', 15)


if __name__ == '__main__':
    D, L = load('Data/Train.txt')
    Dt, Lt = load('Data/Test.txt')
    validation(D, L)



