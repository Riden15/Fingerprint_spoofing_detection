import numpy
import scipy.linalg
from Utility_functions.General_functions import *


# mette tutti i dati in una colonna
def PCA(D, m):
    mu = mcol(D.mean(1))
    C = numpy.dot(D - mu, (D - mu).T) / D.shape[1]
    s, U = numpy.linalg.eigh(C)
    U = U[:, ::-1]
    P = U[:, 0:m]
    return s, P


def SbSw(D, L):
    SB = 0  # between-class variability, variabilità tra le classi
    SW = 0  # within-class variability, variabilità all'interno della classe
    mu = mcol(D.mean(1))
    for i in range(L.max() + 1):
        DCls = D[:, L == i]  # prendo i dati di una sola classe
        muCls = mcol(DCls.mean(1))  # centered data matrix
        SW += numpy.dot(DCls - muCls, (DCls - muCls).T)
        SB += DCls.shape[1] * numpy.dot(muCls - mu, (muCls - mu).T)
    SW /= D.shape[1]  # SW = variazione, quanto sono spreddati i punti per ogni classe
    SB /= D.shape[1]  # SB = distanza delle medie
    return SB, SW


def LDA(D, L, m):
    SB, SW = SbSw(D, L)
    # s = eigenvalues sortati dal più piccolo al più grande
    # U = ogni colonna è un eigenvector del corrispettivo eigenvalue
    s, U = scipy.linalg.eigh(SB, SW)  # in questa funzione va messo sempre passato prima SB e poi SW
    return U[:, ::-1][:, 0:m]  # prendo gli ultimi 2 eigenvector, quelli con eigenvalue più grande
