import numpy

from Validation.validation_MVG import *
from Validation.validation_LR import *
from Validation.Validation_LR_quad import *
from Validation.validation_SVM import *
from Utility_functions.plot_features import *
from Utility_functions.General_functions import *
from Validation.validation_SVM_polynomial import *

def validation(DTR, LTR):
    print("############    MVG    ##############")
    #validation_MVG(DTR,LTR, 5) #FINITO

    print("###########      LR      ##############")
    L = [0.4] # da provare anche per LR normale
    #validation_LR(DTR,LTR, L , 5) # FINITO

    print("############    Quadratic Logistic Regression    ##############")
    validation_quad_LR(DTR, LTR, L,5) # FINITO

    print("############    Support Vector Machine - Linear    ##############")
    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
    #validation_SVM(DTR, LTR, K_arr, C_arr, 'RAW')

    print("############    Support Vector Machine - Quadratic    ##############")
    K_arr = [1., 10.]
    #validation_SVM_polynomial(DTR, LTR, K_arr, 1.0, 'RAW_', [1000])

if __name__ == '__main__':
    D, L = load('Data/Train.txt')
    Dt, Lt = load('Data/Test.txt')
    DTR, LTR = randomize(D, L)
    DTE, LTE = randomize(Dt, Lt)
    #plot_features(DTR, LTR)
    validation(DTR, LTR)



