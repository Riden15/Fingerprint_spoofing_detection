import numpy

from Validation.validation_MVG import *
from Validation.validation_LR import *
from Validation.validation_SVM import *
from Utility_functions.plot_features import *
from Utility_functions.General_functions import *

def validation(DTR, LTR):
    #egnValues, egnVector = PCA_plot(DTR, 10)
    #plot_explained_variance(egnValues)

    print("############    MVG    ##############")
    #validation_MVG(DTR,LTR,155, "MVG, ")
    # con k = 5 i risultati fanno schifo

    print("###########      LR      ##############")
    L = [0.00001]
    #validation_LR(DTR,LTR, L , 'LR, ', 15)

    print("############    Support Vector Machine - Linear    ##############")
    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
    #validation_SVM(DTR, LTR, K_arr, C_arr, 'RAW')


if __name__ == '__main__':
    D, L = load('Data/Train.txt')
    Dt, Lt = load('Data/Test.txt')
    DTR, LTR = randomize(D, L)
    DTE, LTE = randomize(Dt, Lt)
    plot_features(DTR, LTR)
    validation(D, L)



