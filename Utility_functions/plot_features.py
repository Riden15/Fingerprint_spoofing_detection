import sys
import numpy as np

sys.path.append('../')
from Models.Generative_models import *
from Utility_functions.plot_validators import *
from Utility_functions.Validators import *
from prettytable import PrettyTable
from PCA_LDA import *

def compute_correlation(X, Y):
    x_sum = numpy.sum(X)
    y_sum = numpy.sum(Y)

    x2_sum = numpy.sum(X ** 2)
    y2_sum = numpy.sum(Y ** 2)

    sum_cross_prod = numpy.sum(X * Y.T)

    n = X.shape[0]
    numerator = n * sum_cross_prod - x_sum * y_sum
    denominator = numpy.sqrt((n * x2_sum - x_sum ** 2) * (n * y2_sum - y_sum ** 2))

    corr = numerator / denominator
    return corr


def plot_correlations(DTR, title, cmap="Greys"):
    corr = numpy.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            X = DTR[x, :]
            Y = DTR[y, :]
            pearson_elem = compute_correlation(X, Y)
            corr[x][y] = pearson_elem

    sns.set()
    heatmap = sns.heatmap(numpy.abs(corr), linewidth=0.2, cmap=cmap, square=True, cbar=False)
    fig = heatmap.get_figure()
    fig.savefig("./images/" + title + ".svg")

#todo fare un po di refactor a ste funzioni
def plot_features_histograms(DTR, LTR, _title):
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    for i in range(10):
        labels = ["spoofed fingerprint", "fingerprint"]
        title = _title + str(i)
        plt.figure()
        plt.title(title)

        y = DTR[:, LTR == 0][i]
        plt.hist(y, bins=60, density=True, alpha=0.4, linewidth=1.0, color='red', edgecolor='black',
                 label=labels[0])
        y = DTR[:, LTR == 1][i]
        plt.hist(y, bins=60, density=True, alpha=0.4, linewidth=1.0, color='blue', edgecolor='black',
                 label=labels[1])
        plt.legend()
        plt.savefig('./images/hist_' + title + '.svg')
        plt.show()


def plot_PCA(DTR, LTR, m, appendToTitle=''):
    PCA_plot(DTR, LTR, m, appendToTitle + 'PCA_m=' + str(m))


def plot_PCA_LDA(DTR, LTR, m, appendToTitle=''):
    P = PCA_plot(DTR, LTR, m, filename=appendToTitle + 'PCA_m=' + str(m) + ' + LDA', LDA_flag=True)
    DTR = numpy.dot(P.T, -DTR)

    W = LDA(DTR, LTR, 1, m)
    DTR = numpy.dot(W.T, DTR)
    plot_histogram(DTR, LTR, ['spoofed fingerprint', 'fingerprint'], 'PCA_m=' + str(m) + ' + LDA')

def plot_features(DTR, LTR, m=2, appendToTitle=''):
    plot_features_histograms(DTR, LTR, appendToTitle + "feature_")
    plot_correlations(DTR, "heatmap_" + appendToTitle)
    plot_correlations(DTR[:, LTR == 0], "heatmap_spoofedFingerprint_" + appendToTitle, cmap="Reds")
    plot_correlations(DTR[:, LTR == 1], "heatmap_fingerprint_" + appendToTitle, cmap="Blues")
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    plot_PCA(DTR, LTR, m, appendToTitle)
    plot_PCA(DTR, LTR, m, appendToTitle)

    plot_PCA_LDA(DTR, LTR, m, appendToTitle)
    plot_PCA_LDA(DTR, LTR, m, appendToTitle)

def empirical_mean(D):
    return mcol(D.mean(1))


def empirical_covariance(D, mu):
    n = numpy.shape(D)[1]
    DC = D - mcol(mu)
    C = 1 / n * numpy.dot(DC, numpy.transpose(DC))
    return C


def plot_PCA_result(P, D, L, m, filename, LDA_flag):
    plt.figure()
    DP = numpy.dot(P.T, D)
    hlabels = {
        0: "spoofed fingerprint",
        1: "fingerprint"
    }
    if m == 2:
        for i in range(2):
            # I have to invert the sign of the second eigenvector to flip the image
            plt.scatter(DP[:, L == i][0], -DP[:, L == i][1], label=hlabels.get(i), s=10)
            plt.legend()
            plt.tight_layout()
        if LDA_flag is True:
            DTR = numpy.dot(P.T, -D)
            W = LDA(DTR, L, 1) * 100

            plt.quiver(W[0] * -5, W[1] * -5, W[0] * 40, W[1] * 40, units='xy', scale=1, color='g')
            plt.xlim(-65, 65)
            plt.ylim(-25, 25)
        plt.savefig('./images/' + filename + '.png')
        plt.show()
    if m == 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='3d')
        for i in range(2):
            x_vals = DP[:, L == i][0]
            y_vals = -DP[:, L == i][1]
            z_vals = DP[:, L == i][2]

            # I have to invert the sign of the second eigenvector to flip the image
            ax.scatter(x_vals, y_vals, z_vals, label=hlabels.get(i), s=10)
            plt.legend()
            plt.tight_layout()

        if LDA_flag is True:
            DTR = numpy.dot(P.T, -D)
            W = LDA(DTR, L, 1) * 100
            W = W.ravel()
            x = numpy.array([W[0] * -3, W[0] * 3])
            y = numpy.array([W[1] * -3, W[1] * 3])
            z = numpy.array([W[2] * -3, W[2] * 3])
            ax.plot3D(x, y, z)
            ax.view_init(270, 270)

        plt.savefig('./images/' + filename + '.png')
        plt.show()


def PCA_plot(D, L, m=2, filename=None, LDA_flag=False):
    n = numpy.shape(D)[1]
    mu = D.mean(1)
    DC = D - mcol(mu)
    C = 1 / n * numpy.dot(DC, numpy.transpose(DC))
    USVD, s, _ = numpy.linalg.svd(C)
    P = USVD[:, 0:m]

    if filename is not None:
        plot_PCA_result(P, D, L, m, filename, LDA_flag)

    return P


def LDA(D, L, d=1, m=2):
    N = numpy.shape(D)[1]
    mu = D.mean(1)

    tot = 0
    for i in range(m):
        nc = D[:, L == i].shape[1]
        muc = D[:, L == i].mean(1)
        tot += nc * (mcol(muc - mu)).dot(mcol(muc - mu).T)

    SB = 1 / N * tot

    SW = 0
    for i in range(2):
        SW += (L == i).sum() * empirical_covariance(D[:, L == i], empirical_mean(D))

    SW = SW / N

    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:d]
    return W


def gaussianize_features(DTR, TO_GAUSS):
    P = []
    for dIdx in range(DTR.shape[0]):
        DT = mcol(TO_GAUSS[dIdx, :])
        X = DTR[dIdx, :] < DT
        R = (X.sum(1) + 1) / (DTR.shape[1] + 2)
        P.append(scipy.stats.norm.ppf(R))
    return numpy.vstack(P)


def plot_histogram(D, L, labels, title):
    matplotlib.pyplot.figure()
    matplotlib.pyplot.title(title)
    y = D[:, L == 0]
    matplotlib.pyplot.hist(y[0], bins=60, density=True, alpha=0.4, label=labels[0])
    y = D[:, L == 1]
    matplotlib.pyplot.hist(y[0], bins=60, density=True, alpha=0.4, label=labels[1])
    matplotlib.pyplot.legend()
    plt.savefig('./images/hist' + title + '.png')
    matplotlib.pyplot.show()
