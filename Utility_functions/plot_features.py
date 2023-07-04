import sys

sys.path.append('../')
from Models.Generative_models import *
from Utility_functions.plot_validators import *
from Utility_functions.Validators import *
from Models.PCA_LDA import *


# todo fare un po di refactor a ste funzioni: togliere tutto LDA del tipo da qua
def plot_features(DTR, LTR, m=2):
    plot_explained_variance(DTR)
    plot_features_histograms(DTR, LTR, "feature_")
    plot_correlations(DTR, "heatmap_")
    plot_correlations(DTR[:, LTR == 0], "heatmap_spoofedFingerprint_", cmap="Reds")
    plot_correlations(DTR[:, LTR == 1], "heatmap_fingerprint_", cmap="Blues")
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    PCA_LDA_plot(DTR, LTR, m)

    PCA_LDA_hist(DTR, LTR, m)


def plot_explained_variance(DTR):
    egnValues, egnVector = PCA(DTR, 10)
    total_egnValues = sum(egnValues)
    var_exp = [(i / total_egnValues) for i in sorted(egnValues, reverse=True)]

    cum_sum_exp = numpy.cumsum(var_exp)
    # todo migliorare il grafico
    plt.plot(range(0, len(cum_sum_exp)), cum_sum_exp, label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.grid()
    plt.savefig('images/feature_plot/PCA_explainedVariance.png')
    plt.show()


def plot_features_histograms(DTR, LTR, Title):
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    for i in range(10):
        labels = ["spoofed fingerprint", "fingerprint"]
        title = Title + str(i)
        plt.figure()
        plt.title(title)

        y = DTR[:, LTR == 0][i]
        plt.hist(y, bins=60, density=True, alpha=0.4, linewidth=1.0, color='red', edgecolor='black', label=labels[0])
        y = DTR[:, LTR == 1][i]
        plt.hist(y, bins=60, density=True, alpha=0.4, linewidth=1.0, color='blue', edgecolor='black', label=labels[1])
        plt.legend()
        plt.savefig('./images/feature_plot/hist_' + title + '.png')
        plt.show()


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
    fig.savefig("./images/feature_plot/" + title + ".png")


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


def PCA_LDA_plot(DTR, LTR, m=2):
    s, P = PCA(DTR, m=2)
    DTR_PCA = numpy.dot(P.T, -DTR)
    plot_PCA_LDA_result(DTR_PCA, LTR, 'PCA_m=' + str(m), LDA_flag=False)

    W = LDA(DTR_PCA, LTR, 1)
    DTR_PCA_LDA = numpy.dot(W.T, DTR_PCA)
    plot_PCA_LDA_result(DTR_PCA_LDA, LTR, 'PCA_m=' + str(m) + ' + LDA', LDA_flag=True, w=W)


def PCA_LDA_hist(DTR, LTR, m=2):
    s, P = PCA(DTR, m=2)
    DTR_PCA = numpy.dot(P.T, DTR)
    plot_histogram(DTR_PCA, LTR, ['spoofed fingerprint', 'fingerprint'], 'PCA_m=' + str(m))

    W = LDA(DTR_PCA, LTR, 1)
    DTR_PCA_LDA = numpy.dot(W.T, DTR_PCA)
    plot_histogram(DTR_PCA_LDA, LTR, ['spoofed fingerprint', 'fingerprint'], 'PCA_m=' + str(m) + ' + LDA')


def plot_PCA_LDA_result(DTR, LTR, filename, LDA_flag, w=0):
    plt.figure()
    hlabels = {
        0: "spoofed fingerprint",
        1: "fingerprint"
    }
    for i in range(2):
        # I have to invert the sign of the second eigenvector to flip the image
        plt.scatter(DTR[:, LTR == i][0], -DTR[:, LTR == i][1], label=hlabels.get(i), s=10)
        plt.legend()
        plt.tight_layout()
    if LDA_flag is True:
        W = w * 100

        # todo onesto sta roba qua non so da dove esca
        plt.quiver(W[0] * -5, W[1] * -5, W[0] * 40, W[1] * 40, units='xy', scale=1, color='g')
        plt.xlim(-65, 65)
        plt.ylim(-25, 25)
    plt.savefig('./images/feature_plot/' + filename + '.png')
    plt.show()


def plot_histogram(D, L, labels, title):
    matplotlib.pyplot.figure()
    matplotlib.pyplot.title(title)
    y = D[:, L == 0]
    matplotlib.pyplot.hist(y[0], bins=60, density=True, alpha=0.4, label=labels[0])
    y = D[:, L == 1]
    matplotlib.pyplot.hist(y[0], bins=60, density=True, alpha=0.4, label=labels[1])
    matplotlib.pyplot.legend()
    plt.savefig('./images/feature_plot/hist' + title + '.png')
    matplotlib.pyplot.show()


def gaussianize_features(DTR, TO_GAUSS):
    P = []
    for dIdx in range(DTR.shape[0]):
        DT = mcol(TO_GAUSS[dIdx, :])
        X = DTR[dIdx, :] < DT
        R = (X.sum(1) + 1) / (DTR.shape[1] + 2)
        P.append(scipy.stats.norm.ppf(R))
    return numpy.vstack(P)
