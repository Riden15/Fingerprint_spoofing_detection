import numpy
import scipy.special
import scipy.stats as stats
import pylab

from Utility_functions.Validators import confusion_matrix_binary
from Models.Generative_models import *


# todo fare refactor di tutto il file

# ==============================================================================
# ------ BASIC FUNCTIONS --------

def plot_ROC(llrs, LTE, title):
    thresholds = numpy.array(llrs)
    thresholds.sort()
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
    FPR = numpy.zeros(thresholds.size)
    TPR = numpy.zeros(thresholds.size)
    for idx, t in enumerate(thresholds):
        Pred = numpy.int32(llrs > t)
        conf = confusion_matrix_binary(Pred, LTE)
        TPR[idx] = conf[1, 1] / (conf[1, 1] + conf[0, 1])
        FPR[idx] = conf[1, 0] / (conf[1, 0] + conf[0, 0])
    pylab.plot(FPR, TPR)
    pylab.title(title)
    pylab.savefig('./images/ROC_' + title + '.png')
    pylab.show()


# ==============================================================================

# ==============================================================================
# ----------- LOG DENSITIES AND MARGINALS -------------------------------------

def logpdf_GMM(X, gmm):
    S = numpy.zeros((len(gmm), X.shape[1]))

    for g in range(len(gmm)):
        (w, mu, C) = gmm[g]
        S[g, :] = logpdf_GAU_ND_fast(X, mu, C) + numpy.log(w)

    logdens = scipy.special.logsumexp(S, axis=0)

    return S, logdens


# ==============================================================================
# -------------------------------- LBG ----------------------------------------

def LBG(X, alpha, G, psi, typeOf='Full'):
    mu, C = mean_cov_estimate(X)
    U, s, _ = numpy.linalg.svd(C)
    s[s < psi] = psi
    covNew = numpy.dot(U, mcol(s) * U.T)
    GMM = [(1, mu, covNew)]

    while len(GMM) <= G:
        # print('########################################## NEW ITER')
        if len(GMM) != 1:
            if typeOf == 'full':
                GMM = GMM_EM(X, GMM, psi)
            if typeOf == 'diag':
                GMM = GMM_EM_diag(X, GMM, psi)
            if typeOf == 'tied_full':
                GMM = GMM_EM_tied(X, GMM, psi)
            if typeOf == 'tied_diag':
                GMM = GMM_EM_tiedDiag(X, GMM, psi)
        # print('########################################## FIN ITER')
        if len(GMM) == G:
            break

        gmmNew = []
        for i in range(len(GMM)):
            # nuove componenti
            (w, mu, sigma) = GMM[i]
            U, s, vh = numpy.linalg.svd(sigma)
            d = U[:, 0:1] * s[0] ** 0.5 * alpha
            gmmNew.append((w / 2, mu + d, sigma))
            gmmNew.append((w / 2, mu - d, sigma))
            # print("newGmm",gmmNew)
        GMM = gmmNew

    return GMM


# ==============================================================================

# ==============================================================================
# ---------------------- GMM _ EM, GMM _ FULL DIAGONAL,K FOLD -----------------
def GMM_EM(X, gmm, psi=0.01):
    '''
    EM algorithm for GMM full covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    If psi is given it's used for constraining the eigenvalues of the
    covariance matrices to be larger or equal to psi
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]

    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdf_GMM(X, gmm)
        llNew = SM.sum() / N
        P = numpy.exp(SJ - SM)
        gmmNew = []
        for g in range(G):
            gamma = P[g, :]
            Z = gamma.sum()
            F = (vrow(gamma) * X).sum(1)
            S = numpy.dot(X, (vrow(gamma) * X).T)
            w = Z / N
            mu = mcol(F / Z)
            Sigma = S / Z - numpy.dot(mu, mu.T)
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s < psi] = psi
            Sigma = numpy.dot(U, mcol(s) * U.T)
            gmmNew.append((w, mu, Sigma))
        gmm = gmmNew
        # print(llNew)
    # print(llNew-llOld)
    return gmm


def GMM_EM_tiedDiag(X, gmm, psi=0.01):  # X -> ev
    '''
    EM algorithm for GMM tied diagonal covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    If psi is given it's used for constraining the eigenvalues of the
    covariance matrices to be larger or equal to psi
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    sigma_array = []
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ = numpy.zeros((G, N))
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND_fast(
                X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        SM = scipy.special.logsumexp(SJ, axis=0)
        llNew = SM.sum() / N
        P = numpy.exp(SJ - SM)
        gmmNew = []
        sigmaTied = numpy.zeros((X.shape[0], X.shape[0]))
        for g in range(G):
            # m step
            gamma = P[g, :]
            Z = gamma.sum()
            F = (vrow(gamma) * X).sum(1)
            S = numpy.dot(X, (vrow(gamma) * X).T)
            w = Z / N
            mu = mcol(F / Z)
            # Sigma = S/Z - numpy.dot(mu, mu.T)
            # Sigma = Sigma * numpy.eye(Sigma.shape[0])
            # Sigma = Sigma * Z
            # gmmNew.append((w, mu, Sigma))
            sigma = S / Z - numpy.dot(mu, mu.T)
            sigmaTied += Z * sigma
            gmmNew.append((w, mu))

            # calculate tied covariance
        gmm = gmmNew
        sigmaTied /= N
        sigmaTied *= numpy.eye(sigma.shape[0])
        U, s, _ = numpy.linalg.svd(sigmaTied)
        s[s < psi] = psi
        sigmaTied = numpy.dot(U, mcol(s) * U.T)

        newGmm = []
        for i in range(len(gmm)):
            (w, mu) = gmm[i]
            newGmm.append((w, mu, sigmaTied))

        gmm = newGmm
        # print(llNew,'llnew') #increase - if decrease problem
    # print(llNew-llOld,'llnew-llold')
    return gmm


def GMM_EM_tied(X, gmm, psi=0.01):
    '''
    EM algorithm for GMM tied full covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    If psi is given it's used for constraining the eigenvalues of the
    covariance matrices to be larger or equal to psi
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    sigma_array = []
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdf_GMM(X, gmm)
        llNew = SM.sum() / N
        P = numpy.exp(SJ - SM)
        gmmNew = []

        sigmaTied = numpy.zeros((X.shape[0], X.shape[0]))
        for g in range(G):
            # m step
            gamma = P[g, :]
            Z = gamma.sum()
            F = (vrow(gamma) * X).sum(1)
            S = numpy.dot(X, (vrow(gamma) * X).T)
            w = Z / N
            mu = mcol(F / Z)
            Sigma = S / Z - numpy.dot(mu, mu.T)
            sigmaTied += Z * Sigma
            gmmNew.append((w, mu))
            # Sigma = Sigma * Z
            # gmmNew.append((w, mu, Sigma))

        # calculate tied covariance
        gmm = gmmNew
        sigmaTied /= N
        U, s, _ = numpy.linalg.svd(sigmaTied)
        s[s < psi] = psi
        sigmaTied = numpy.dot(U, mcol(s) * U.T)

        gmmNew = []
        for g in range(G):
            (w, mu) = gmm[g]
            gmmNew.append((w, mu, sigmaTied))
        gmm = gmmNew
        # print(llNew,'llnew') #increase - if decrease problem
    # print(llNew-llOld,'llnew-llold')
    return gmm


def GMM_Full(DTR, DTE, LTR, alpha, G, typeOf, psi=0.01):
    DTR0 = DTR[:, LTR == 0]
    gmm0 = LBG(DTR0, alpha, G, psi, typeOf)
    _, llr0 = logpdf_GMM(DTE, gmm0)

    DTR1 = DTR[:, LTR == 1]
    gmm1 = LBG(DTR1, alpha, G, psi, typeOf)
    _, llr1 = logpdf_GMM(DTE, gmm1)

    return llr1 - llr0


def GMM_EM_diag(X, gmm, psi=0.01):
    '''
    EM algorithm for GMM diagonal covariance
    It estimates the parameters of a GMM that maximize the ll for
    a training set X
    If psi is given it's used for constraining the eigenvalues of the
    covariance matrices to be larger or equal to psi
    '''
    llNew = None
    llOld = None
    G = len(gmm)
    N = X.shape[1]
    while llOld is None or llNew - llOld > 1e-6:
        llOld = llNew
        SJ, SM = logpdf_GMM(X, gmm)
        llNew = SM.sum() / N
        P = numpy.exp(SJ - SM)

        gmmNew = []
        for g in range(G):
            # m step
            gamma = P[g, :]
            Z = gamma.sum()
            F = (vrow(gamma) * X).sum(1)
            S = numpy.dot(X, (vrow(gamma) * X).T)
            w = Z / N
            mu = mcol(F / Z)
            Sigma = S / Z - numpy.dot(mu, mu.T)
            Sigma = Sigma * numpy.eye(Sigma.shape[0])
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s < psi] = psi
            sigma = numpy.dot(U, mcol(s) * U.T)
            gmmNew.append((w, mu, sigma))
        gmm = gmmNew
        # print(llNew)
    # print(llNew)
    return gmm
