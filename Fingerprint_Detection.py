import numpy
import matplotlib
import matplotlib.pyplot as plt

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

def kFold(D, L):
    k = D.shape[1] # numero di sample
    splits,labels = split_data(D, L, k)
    mvgAcc = 0
    nbgcAcc = 0
    tcgcAcc = 0
    tnbcAcc = 0

    for i in range(len(splits)):
        DTE = splits[i]
        LTE = labels[i]
        if i == 0:
            DTR = numpy.hstack(splits[i+1:])
            LTR = numpy.hstack(labels[i+1:])
        elif i == k-1:
            DTR = numpy.hstack(splits[:i])
            LTR = numpy.hstack(labels[:i])
        else:
            DTR = numpy.hstack(numpy.vstack((splits[:i], splits[i + 1:])))
            LTR = numpy.hstack(numpy.vstack((labels[:i], labels[i + 1:])))

        acc = Gaussian_classify(DTR,LTR,DTE,LTE)
        mvgAcc+=acc
        acc = Naive_Bayes_Gaussian_classify(DTR,LTR,DTE,LTE)
        nbgcAcc+=acc
        acc = Tied_Covariance_Gaussian_classifier(DTR,LTR,DTE,LTE)
        tcgcAcc+=acc
        acc = Tied_Naive_Covariance_Gaussian_classifier(DTR, LTR, DTE, LTE)
        tnbcAcc += acc

if __name__ == '__main__':
    D, L = load('Data/Train.txt')
    print(D)