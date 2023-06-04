import numpy
from Models import Generative_models
import PCA_LDA

def mcol(v):
    return v.reshape((v.size, 1))

def split_data(D,L,k):
    return numpy.array(numpy.hsplit(D,k)) , numpy.array(numpy.array_split(L,k))

# accArray[0] = accuracy Gaussian Classify
# accArray[1] = accuracy log Gaussian Classify
# accArray[2] = accuracy Naive Bayes Gaussian Classify
# accArray[3] = accuracy Tied Gaussian Classify
# accArray[4] = accuracy Tied Naive Gaussian Classify
# accArray[5] = accuracy Linear Regression
def kFold(D, L, k):
    splits,labels = split_data(D, L, k)
    accArray = [0,0,0,0,0,0]

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

        #acc ritorna 1 se ha azzeccato e 0 se ha sbagliato.
        #mvgAcc in questo caso è 146 --> di 150 sample ha sbagliato a predirne 4
        acc = Generative_models.Gaussian_classify(DTR, LTR, DTE, LTE)
        accArray[0]+=acc
        acc = Generative_models.Gaussian_classify_log(DTR, LTR, DTE, LTE)
        accArray[1]+=acc
        acc = Generative_models.Naive_Bayes_Gaussian_classify(DTR, LTR, DTE, LTE)
        accArray[2]+=acc
        acc = Generative_models.Tied_Covariance_Gaussian_classifier(DTR, LTR, DTE, LTE)
        accArray[3]+=acc
        acc = Generative_models.Tied_Naive_Covariance_Gaussian_classifier(DTR, LTR, DTE, LTE)
        accArray[4]+=acc
        #acc = lr_binary(1, DTR, LTR, DTE, LTE)
        #accArray[5]+=acc

    return [val/L.size for val in accArray]

def Test_kFold_with_optimal_number_of_PC(D, L, k):

    # Queste matrici contengono le accuracy di ogni modello per ogni numero possibile di dimensioni
    # ridotto da PCA e da LDA. Le colonne dividono i modelli, le righe le dimensioni
    matrixPCA = numpy.zeros((11, 6))
    matrixPCA_LDA = numpy.zeros((11, 6))
    for i in range(1,11):
        P = PCA_LDA.PCA(D, i)
        D_PCA = (numpy.dot(P.T, D))
        matrixPCA[i] = kFold(D_PCA, L, k)

        P2 = PCA_LDA.LDA1(D_PCA, L, i)
        D_PCA_LDA = numpy.dot(P2.T, D_PCA)
        matrixPCA_LDA[i]=kFold(D_PCA_LDA, L, k)

    maxPCA = numpy.argmax(matrixPCA, axis=0)  # valori ottimali di dimensioni usando solo il PCA per ogni modello
    maxLDA = numpy.argmax(matrixPCA_LDA, axis=0)   # valori ottimali di dimensioni usando PCA ed LDA per ogni modello

    print("kFold Test for Gaussian Classifier using PCA with optimal number of principal component: number of dimension=" + str(maxPCA[0]) + ", accuracy=" + str((matrixPCA[maxPCA[0], 0])*100))
    print("kFold Test for Log Gaussian Classifier using PCA with optimal number of principal component: number of dimension=" + str(maxPCA[1]) + ", accuracy=" + str((matrixPCA[maxPCA[1], 1])*100))
    print("kFold Test for Naive Bayes Gaussian Classifier using PCA with optimal number of principal component: number of dimension=" + str(maxPCA[2]) + ", accuracy=" + str((matrixPCA[maxPCA[2], 2])*100))
    print("kFold Test for Tied Covariance Gaussian Classifier using PCA with optimal number of principal component: number of dimension=" + str(maxPCA[3]) + ", accuracy=" + str((matrixPCA[maxPCA[3], 3])*100))
    print("kFold Test for Tied Naive Bayes Gaussian Classifier using PCA with optimal number of principal component: number of dimension=" + str(maxPCA[4]) + ", accuracy=" + str((matrixPCA[maxPCA[4], 4])*100))
    print("kFold Test for Logistic Regression using PCA with optimal number of principal component: number of dimension=" + str(maxPCA[5]) + ", accuracy=" + str((matrixPCA[maxPCA[5], 5])*100))

    print("---------------------------------------------------------------")

    print("kFold Test for Gaussian Classifier using PCA and LDA with optimal number of principal component: number of dimension=" + str(maxLDA[0]) + ", accuracy=" + str((matrixPCA_LDA[maxLDA[0], 0])*100))
    print("kFold Test for Log Gaussian Classifier using PCA and LDA with optimal number of principal component: number of dimension=" + str(maxLDA[1]) + ", accuracy=" + str((matrixPCA_LDA[maxLDA[1], 1])*100))
    print("kFold Test for Naive Bayes Gaussian Classifier using PCA and LDA with optimal number of principal component: number of dimension=" + str(maxLDA[2]) + ", accuracy=" + str((matrixPCA_LDA[maxLDA[2], 2])*100))
    print("kFold Test for Tied Covariance Gaussian Classifier using PCA and LDA with optimal number of principal component: number of dimension=" + str(maxLDA[3]) + ", accuracy=" + str((matrixPCA_LDA[maxLDA[3], 3])*100))
    print("kFold Test for Tied Naive Bayes Gaussian Classifier using PCA and LDA with optimal number of principal component: number of dimension=" + str(maxLDA[4]) + ", accuracy=" + str((matrixPCA_LDA[maxLDA[4], 4])*100))
    print("kFold Test for Logistic Regression using PCA and LDA with optimal number of principal component: number of dimension=" + str(maxLDA[5]) + ", accuracy=" + str((matrixPCA_LDA[maxLDA[5], 5])*100))

# ritorna un array con le accuracy di ogni modello provato
# accArray[0] = accuracy Gaussian Classify
# accArray[1] = accuracy log Gaussian Classify
# accArray[2] = accuracy Naive Bayes Gaussian Classify
# accArray[3] = accuracy Tied Gaussian Classify
# accArray[4] = accuracy Tied Naive Gaussian Classify
# accArray[5] = accuracy Linear Regression
def split_db_and_try_models(D, L):

    accArray = []
    (DTR, LTR), (DTE, LTE) = Generative_models.split_db_2to1(D, L)
    acc = Generative_models.Gaussian_classify(DTR, LTR, DTE, LTE)
    accArray.append(acc)
    acc = Generative_models.Gaussian_classify_log(DTR, LTR, DTE, LTE)
    accArray.append(acc)
    acc = Generative_models.Naive_Bayes_Gaussian_classify(DTR, LTR, DTE, LTE)
    accArray.append(acc)
    acc = Generative_models.Tied_Covariance_Gaussian_classifier(DTR, LTR, DTE, LTE)
    accArray.append(acc)
    acc = Generative_models.Tied_Naive_Covariance_Gaussian_classifier(DTR, LTR, DTE, LTE)
    accArray.append(acc)
    acc = Generative_models.lr_binary(1, DTR, LTR, DTE, LTE)
    accArray.append(acc)

    return accArray

#funzione che prova i modelli per ogni possibile numero di dimensioni
# per ogni modello si creano 10 diversi modello ognuno con un hyperparameter del PCA diverso per capire il migliore
def Test_split_with_optimal_number_of_PC(D, L):

    # Queste matrici contengono le accuracy di ogni modello per ogni numero possibile di dimensioni
    # ridotto da PCA e da LDA. Le colonne dividono i modelli, le righe le dimensioni
    matrixPCA = numpy.zeros((11, 6))
    matrixPCA_LDA = numpy.zeros((11, 6))
    for i in range(1,11):
        P = PCA_LDA.PCA(D, i)
        D_PCA = (numpy.dot(P.T, D))
        matrixPCA[i] = split_db_and_try_models(D_PCA, L)

        P2 = PCA_LDA.LDA1(D_PCA, L, i)
        D_PCA_LDA = numpy.dot(P2.T, D_PCA)
        matrixPCA_LDA[i]=split_db_and_try_models(D_PCA_LDA, L)

    maxPCA = numpy.argmax(matrixPCA, axis=0)  # valori ottimali di dimensioni usando solo il PCA per ogni modello
    maxLDA = numpy.argmax(matrixPCA_LDA, axis=0)   # valori ottimali di dimensioni usando PCA ed LDA per ogni modello

    print("Test for Gaussian Classifier using PCA with optimal number of principal component: number of dimension=" + str(maxPCA[0]) + ", accuracy=" + str((matrixPCA[maxPCA[0], 0])*100))
    print("Test for Log Gaussian Classifier using PCA with optimal number of principal component: number of dimension=" + str(maxPCA[1]) + ", accuracy=" + str((matrixPCA[maxPCA[1], 1])*100))
    print("Test for Naive Bayes Gaussian Classifier using PCA with optimal number of principal component: number of dimension=" + str(maxPCA[2]) + ", accuracy=" + str((matrixPCA[maxPCA[2], 2])*100))
    print("Test for Tied Covariance Gaussian Classifier using PCA with optimal number of principal component: number of dimension=" + str(maxPCA[3]) + ", accuracy=" + str((matrixPCA[maxPCA[3], 3])*100))
    print("Test for Tied Naive Bayes Gaussian Classifier using PCA with optimal number of principal component: number of dimension=" + str(maxPCA[4]) + ", accuracy=" + str((matrixPCA[maxPCA[4], 4])*100))
    print("Test for Logistic Regression using PCA with optimal number of principal component: number of dimension=" + str(maxPCA[5]) + ", accuracy=" + str((matrixPCA[maxPCA[5], 5])*100))

    print("---------------------------------------------------------------")

    print("Test for Gaussian Classifier using PCA and LDA with optimal number of principal component: number of dimension=" + str(maxLDA[0]) + ", accuracy=" + str((matrixPCA_LDA[maxLDA[0], 0])*100))
    print("Test for Log Gaussian Classifier using PCA and LDA with optimal number of principal component: number of dimension=" + str(maxLDA[1]) + ", accuracy=" + str((matrixPCA_LDA[maxLDA[1], 1])*100))
    print("Test for Naive Bayes Gaussian Classifier using PCA and LDA with optimal number of principal component: number of dimension=" + str(maxLDA[2]) + ", accuracy=" + str((matrixPCA_LDA[maxLDA[2], 2])*100))
    print("Test for Tied Covariance Gaussian Classifier using PCA and LDA with optimal number of principal component: number of dimension=" + str(maxLDA[3]) + ", accuracy=" + str((matrixPCA_LDA[maxLDA[3], 3])*100))
    print("Test for Tied Naive Bayes Gaussian Classifier using PCA and LDA with optimal number of principal component: number of dimension=" + str(maxLDA[4]) + ", accuracy=" + str((matrixPCA_LDA[maxLDA[4], 4])*100))
    print("Test for Logistic Regression using PCA and LDA with optimal number of principal component: number of dimension=" + str(maxLDA[5]) + ", accuracy=" + str((matrixPCA_LDA[maxLDA[5], 5])*100))

