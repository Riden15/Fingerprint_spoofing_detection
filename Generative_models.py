import numpy
import scipy
import matplotlib
import matplotlib.pyplot as plt
import PCA_LDA

# mette tutti i dati in una colonna
def mcol(v):
    return v.reshape((v.size, 1))

# mette tutti i dati in una riga
def vrow(v):
    return v.reshape((1, v.size))

#calcolo log likelihood
def logpdf_GAU_ND_fast(X, mu, C):
    XC = X - mu
    M = X.shape[0]
    const = - 0.5 * M * numpy.log(2*numpy.pi)
    logdet = numpy.linalg.slogdet(C)[1]
    L = numpy.linalg.inv(C)
    v = (XC * numpy.dot(L, XC)).sum(0)
    return const - 0.5 * logdet - 0.5 * v

def mean_cov_estimate(X):
    mu = mcol(X.mean(1))
    C = numpy.dot(X-mu, (X-mu).T)/X.shape[1]
    return mu, C

# serve per il Naive Bayes
def mean_covDiagonal_estimate(X):
    mu = mcol(X.mean(1))
    C = numpy.dot(X-mu, (X-mu).T)/X.shape[1]
    n = C.shape[1]
    Mid = numpy.identity(n)
    Cd = numpy.multiply(Mid, C)
    return mu, Cd

# per il Tied Covariance
# deve ritornare la media per la classe passata (Ds sono i dati di una sola classe)
# e la within class covariance e servono i dati di tutte le classi
def mean_Sw(D, L, Ds):  
    SW = 0  # within-class variability, variabilità all'interno della classe
    mu = mcol(Ds.mean(1))
    for i in range(L.max()+1):
        DCls = D[:, L==i]  # prendo i dati di una sola classe
        muCls = mcol(DCls.mean(1))
        SW += numpy.dot(DCls-muCls, (DCls-muCls).T)
    SW /= D.shape[1] # SW = variazione, quanto sono spreddati i punti per ogni classe
    return mu, SW

# per il Tied Naive Bayes
# l'unica cosa che cambia dal Tied normale è che si prende solo la diagonal della within class covariance matrix
def mean_Sw_Diagonal(D, L, Ds):  
    SW = 0  # within-class variability, variabilità all'interno della classe
    mu = mcol(Ds.mean(1))
    for i in range(L.max()+1):
        DCls = D[:, L==i]  # prendo i dati di una sola classe
        muCls = mcol(DCls.mean(1)) # centered data matrix
        SW += numpy.dot(DCls-muCls, (DCls-muCls).T)
    SW /= D.shape[1] # SW = variazione, quanto sono spreddati i punti per ogni classe
    n = SW.shape[1]
    Mid = numpy.identity(n)
    SWdiag = numpy.multiply(Mid, SW)
    return mu, SWdiag

# splitta il dataset in 2, 2/3 sono per il training e 1/3 per i test
def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)


## CLASSIFIERS ##

def Gaussian_classify_prova(DTR,LTR):
    hCLs = {}
    for lab in [0,1]:
        DCLS = DTR[:, LTR==lab]
        hCLs[lab] = mean_cov_estimate(DCLS)
    return hCLs

def Test(hCLs, DTV, LTV):
    prior = mcol(numpy.ones(2) / 2.0)
    S = []
    for hyp in [0, 1]:
        mu, C = hCLs[hyp]
        fcond = numpy.exp(logpdf_GAU_ND_fast(DTV, mu, C))
        S.append(vrow(fcond))
    S = numpy.vstack(S)
    S = S * prior
    P = S / vrow(S.sum(0))

    # CALCOLO ACCURACY
    max = numpy.argmax(P, axis=0)
    prediction = [max == LTV]
    prediction = numpy.vstack(prediction)
    numPrediction = prediction.sum(0).sum(0)
    accuracy = numPrediction / LTV.shape[0]
    return accuracy

def Gaussian_classify(DTR,LTR,DTV,LTV):
    hCLs = {}
    for lab in [0,1]:
        DCLS = DTR[:, LTR==lab]
        hCLs[lab] = mean_cov_estimate(DCLS)

    # ClASSIFICATION
    prior = mcol(numpy.ones(2)/2.0)
    S = []
    for hyp in [0,1]:
        mu, C = hCLs[hyp]
        fcond = numpy.exp(logpdf_GAU_ND_fast(DTV, mu, C))
        S.append(vrow(fcond))
    S = numpy.vstack(S)
    S = S * prior
    P = S / vrow(S.sum(0))
    
    # CALCOLO ACCURACY
    max = numpy.argmax(P, axis=0)
    prediction = [max==LTV]
    prediction = numpy.vstack(prediction)
    numPrediction = prediction.sum(0).sum(0)
    accuracy = numPrediction/LTV.shape[0]
    return accuracy

def Gaussian_classify_log(DTR,LTR,DTV,LTV):
    hCls = {}
    for lab in [0,1]:
        DCLS = DTR[:, LTR==lab]
        hCls[lab] = mean_cov_estimate(DCLS)

    # CLASSIFICATION
    logprior = numpy.log(mcol(numpy.ones(2)/2.0))
    S = []
    for hyp in [0,1]:
        mu, C = hCls[hyp]
        fcond = logpdf_GAU_ND_fast(DTV, mu, C)
        S.append(vrow(fcond))
    S = numpy.vstack(S)
    S = S + logprior
    logP = S - vrow(scipy.special.logsumexp(S, 0))
    P = numpy.exp(logP)

    max = numpy.argmax(P, axis=0)
    prediction = [max==LTV] 
    prediction = numpy.vstack(prediction)
    numPrediction = prediction.sum(0).sum(0)
    accuracy = numPrediction/LTV.shape[0]
    return accuracy
    
def Naive_Bayes_Gaussian_classify(DTR,LTR,DTV,LTV):
    hCLs = {}
    for lab in [0,1]:
        DCLS = DTR[:, LTR==lab]
        hCLs[lab] = mean_covDiagonal_estimate(DCLS)

    # ClASSIFICATION
    prior = mcol(numpy.ones(2)/2.0)
    S = []
    for hyp in [0,1]:
        mu, C = hCLs[hyp] 
        fcond = numpy.exp(logpdf_GAU_ND_fast(DTV, mu, C)) 
        S.append(vrow(fcond)) 

    S = numpy.vstack(S)
    S = S * prior
    P = S / vrow(S.sum(0))
    max = numpy.argmax(P, axis=0)
    prediction = [max==LTV]  
    prediction = numpy.vstack(prediction)
    numPrediction = prediction.sum(0).sum(0)
    accuracy = numPrediction/LTV.shape[0]
    return accuracy

def Tied_Covariance_Gaussian_classifier(DTR,LTR,DTV,LTV):
    hCLs = {}
    for lab in [0,1]:
        DCLS = DTR[:, LTR==lab]
        hCLs[lab] = mean_Sw(DTR, LTR, DCLS)
    
    # ClASSIFICATION
    prior = mcol(numpy.ones(2)/2.0)
    S = []
    for hyp in [0,1]:
        mu, C = hCLs[hyp] 
        fcond = numpy.exp(logpdf_GAU_ND_fast(DTV, mu, C)) 
        S.append(vrow(fcond)) 
    
    S = numpy.vstack(S)
    S = S * prior
    P = S / vrow(S.sum(0))
    max = numpy.argmax(P, axis=0)
    prediction = [max==LTV]  
    prediction = numpy.vstack(prediction)
    numPrediction = prediction.sum(0).sum(0)
    accuracy = numPrediction/LTV.shape[0]
    return accuracy

def Tied_Naive_Covariance_Gaussian_classifier(DTR,LTR,DTV,LTV):
    hCLs = {}
    for lab in [0,1]:
        DCLS = DTR[:, LTR==lab]
        hCLs[lab] = mean_Sw_Diagonal(DTR, LTR, DCLS)
    
    # ClASSIFICATION
    prior = mcol(numpy.ones(2)/2.0)
    S = []
    for hyp in [0,1]:
        mu, C = hCLs[hyp] 
        fcond = numpy.exp(logpdf_GAU_ND_fast(DTV, mu, C)) 
        S.append(vrow(fcond)) 
    
    S = numpy.vstack(S)
    S = S * prior
    P = S / vrow(S.sum(0))
    max = numpy.argmax(P, axis=0)
    prediction = [max==LTV]  
    prediction = numpy.vstack(prediction)
    numPrediction = prediction.sum(0).sum(0)
    accuracy = numPrediction/LTV.shape[0]
    return accuracy


def logreg_obj(v, DTR, LTR, l):
    w = v[0:-1]
    b = v[-1]
    first = l / 2 * numpy.power(numpy.linalg.norm(w), 2)
    second = 1 / DTR.shape[1]
    add = 0
    for i in range(DTR.shape[1]):
        z = 2 * LTR[i] - 1
        add += numpy.logaddexp(0, -z * (numpy.dot(w.T, DTR[:, i]) + b))
    return first + second * add

def lr_binary(l, DTR, LTR, DTE, LTE):
    param = numpy.zeros(DTR.shape[0] + 1)

    logreg_obj(param, DTR, LTR, l)
    x, d, f = scipy.optimize.fmin_l_bfgs_b(logreg_obj, param, approx_grad=True, args=(DTR, LTR, l))

    w = x[0:-1]  # tutti tranne l'ultimo
    b = x[-1]  # ultimo valore di x
    S = numpy.dot(w.T, DTE) + b
    predicted_label = numpy.zeros(S.size, dtype=int)
    for i in range(S.size):
        if S[i] > 0:
            predicted_label.put(i, 1)
        else:
            predicted_label.put(i, 0)

    equal = predicted_label == LTE
    correct = sum(equal)
    accuracy = correct / LTE.size
    #error_rate = 1 - accuracy
    return accuracy


