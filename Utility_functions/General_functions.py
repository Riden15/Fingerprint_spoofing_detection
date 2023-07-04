import sys
import numpy

def mcol(v):
    return v.reshape((v.size, 1))

def vrow(vec):
    return vec.reshape((1,vec.shape[0]))

def randomize(D, L, seed=0):
    nTrain = int(D.shape[1])
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]

    DTR = D[:, idxTrain]
    LTR = L[idxTrain]

    return DTR, LTR

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

def kfold(fold, k, FoldedData_List, FoldedLabel_List):
    Dtr = []
    Ltr = []
    if fold == 0:
        Dtr.append(numpy.hstack(FoldedData_List[fold + 1:]))
        Ltr.append(numpy.hstack(FoldedLabel_List[fold + 1:]))
    elif fold == k - 1:
        Dtr.append(numpy.hstack(FoldedData_List[:fold]))
        Ltr.append(numpy.hstack(FoldedLabel_List[:fold]))
    else:
        Dtr.append(numpy.hstack(FoldedData_List[:fold]))
        Dtr.append(numpy.hstack(FoldedData_List[fold + 1:]))
        Ltr.append(numpy.hstack(FoldedLabel_List[:fold]))
        Ltr.append(numpy.hstack(FoldedLabel_List[fold + 1:]))

    Dtr = numpy.hstack(Dtr)  # fold selezionati per training (dati)
    Ltr = numpy.hstack(Ltr)  # fold selezionati per training (label)
    Dte = FoldedData_List[fold]  # singolo fold selezionato per evaluation (dati)
    Lte = FoldedLabel_List[fold]  # singolo fold selezionato per evaluation (label)
    return Dtr, Ltr, Dte, Lte
