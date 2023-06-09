from Utility_functions.General_functions import *


# calcolo log likelihood
def logpdf_GAU_ND_fast(X, mu, C):
    XC = X - mu
    M = X.shape[0]
    const = - 0.5 * M * numpy.log(2 * numpy.pi)
    logdet = numpy.linalg.slogdet(C)[1]
    L = numpy.linalg.inv(C)
    v = (XC * numpy.dot(L, XC)).sum(0)
    return const - 0.5 * logdet - 0.5 * v


def mean_cov_estimate(X):
    mu = mcol(X.mean(1))
    C = numpy.dot(X - mu, (X - mu).T) / X.shape[1]
    return mu, C


# serve per il Naive Bayes
def mean_covDiagonal_estimate(X):
    mu = mcol(X.mean(1))
    C = numpy.dot(X - mu, (X - mu).T) / X.shape[1]
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
    for i in range(L.max() + 1):
        DCls = D[:, L == i]  # prendo i dati di una sola classe
        muCls = mcol(DCls.mean(1))
        SW += numpy.dot(DCls - muCls, (DCls - muCls).T)
    SW /= D.shape[1]  # SW = variazione, quanto sono spreddati i punti per ogni classe
    return mu, SW


# per il Tied Naive Bayes
# l'unica cosa che cambia dal Tied normale è che si prende solo la diagonal della within class covariance matrix
def mean_Sw_Diagonal(D, L, Ds):
    SW = 0  # within-class variability, variabilità all'interno della classe
    mu = mcol(Ds.mean(1))
    for i in range(L.max() + 1):
        DCls = D[:, L == i]  # prendo i dati di una sola classe
        muCls = mcol(DCls.mean(1))  # centered data matrix
        SW += numpy.dot(DCls - muCls, (DCls - muCls).T)
    SW /= D.shape[1]  # SW = variazione, quanto sono spreddati i punti per ogni classe
    n = SW.shape[1]
    Mid = numpy.identity(n)
    SWdiag = numpy.multiply(Mid, SW)
    return mu, SWdiag


# splitta il dataset in 2, 2/3 sono per il training e 1/3 per i test
def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
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

def MVG(DTR, LTR, DTE):
    hCLs = {}
    for lab in [0, 1]:
        DCLS = DTR[:, LTR == lab]
        hCLs[lab] = mean_cov_estimate(DCLS)

    # ClASSIFICATION
    prior = mcol(numpy.ones(2) / 2.0)
    S = []
    for hyp in [0, 1]:
        mu, C = hCLs[hyp]
        fcond = numpy.exp(logpdf_GAU_ND_fast(DTE, mu, C))
        S.append(vrow(fcond))
    ll = numpy.vstack(S)
    S = numpy.vstack(S)
    S = S * prior
    P = S / vrow(S.sum(0))
    return numpy.log(ll[1] / ll[0])


def Naive_Bayes_Gaussian_classify(DTR, LTR, DTE):
    hCLs = {}
    for lab in [0, 1]:
        DCLS = DTR[:, LTR == lab]
        hCLs[lab] = mean_covDiagonal_estimate(DCLS)

    # ClASSIFICATION
    prior = mcol(numpy.ones(2) / 2.0)
    S = []
    for hyp in [0, 1]:
        mu, C = hCLs[hyp]
        fcond = numpy.exp(logpdf_GAU_ND_fast(DTE, mu, C))
        S.append(vrow(fcond))
    ll = numpy.vstack(S)
    S = numpy.vstack(S)
    S = S * prior
    P = S / vrow(S.sum(0))
    return numpy.log(ll[1] / ll[0])


def Tied_Covariance_Gaussian_classifier(DTR, LTR, DTE):
    hCLs = {}
    for lab in [0, 1]:
        DCLS = DTR[:, LTR == lab]
        hCLs[lab] = mean_Sw(DTR, LTR, DCLS)

    # ClASSIFICATION
    prior = mcol(numpy.ones(2) / 2.0)
    S = []
    for hyp in [0, 1]:
        mu, C = hCLs[hyp]
        fcond = numpy.exp(logpdf_GAU_ND_fast(DTE, mu, C))
        S.append(vrow(fcond))

    ll = numpy.vstack(S)
    S = numpy.vstack(S)
    S = S * prior
    P = S / vrow(S.sum(0))
    return numpy.log(ll[1] / ll[0])


def Tied_Naive_Covariance_Gaussian_classifier(DTR, LTR, DTE):
    hCLs = {}
    for lab in [0, 1]:
        DCLS = DTR[:, LTR == lab]
        hCLs[lab] = mean_Sw_Diagonal(DTR, LTR, DCLS)

    # ClASSIFICATION
    prior = mcol(numpy.ones(2) / 2.0)
    S = []
    for hyp in [0, 1]:
        mu, C = hCLs[hyp]
        fcond = numpy.exp(logpdf_GAU_ND_fast(DTE, mu, C))
        S.append(vrow(fcond))

    ll = numpy.vstack(S)
    S = numpy.vstack(S)
    S = S * prior
    P = S / vrow(S.sum(0))
    return numpy.log(ll[1] / ll[0])
