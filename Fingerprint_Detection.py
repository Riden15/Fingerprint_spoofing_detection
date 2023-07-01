import numpy

from Validation.validation_GMM import validation_GMM_tot
from Validation.validation_MVG import *
from Validation.validation_LR import *
from Validation.Validation_LR_quad import *
from Validation.validation_SVM import *
from Utility_functions.plot_features import *
from Utility_functions.General_functions import *
from Validation.validation_SVM_RBF import validation_SVM_RBF
from Validation.validation_SVM_polynomial import *

def validation(DTR, LTR):
    print("############    MVG    ##############")
    #validation_MVG(DTR,LTR, 5) #FINITO

    print("###########      LR      ##############")
    L = [0.4] # da provare anche per LR normale, 0.4 è l'ottimale per il quadratic
    #validation_LR(DTR,LTR, L , 5) # FINITO

    print("############    Quadratic Logistic Regression    ##############")
    #validation_quad_LR(DTR, LTR, L,5) # DA TOGLIERE, NON ANDAVA FATTO

    print("############    Support Vector Machine - Linear    ##############")
    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
    #validation_SVM(DTR, LTR, K_arr, C_arr, 5) # FINITO

    print("############    Support Vector Machine - Quadratic    ##############")
    K_arr = [0.1, 1, 10]
    C_arr = [0.01, 0.1, 1.0, 10]
    constant = [0, 1]
    #validation_SVM_polynomial(DTR, LTR, K_arr, C_arr, constant, 5)

    print("############    Support Vector Machine - RBF    ##############")
    K_arr = [0.1, 1, 10]
    C_arr = [1, 10, 100]
    gamma_arr=[0.001, 0.0001]
    #validation_SVM_RBF(DTR, LTR, K_arr, gamma_arr, C_arr, 5)

    print("############    Gaussian Mixture Models   ##############")
    validation_GMM_tot(DTR, LTR, 5)
    #validation_GMM_ncomp(DTR, LTR, 0.5, 2)
    #validation_GMM_ncomp(DTR, LTR, 0.1, 2)
    #validation_GMM_ncomp(DTR, LTR, 0.9, 2)


if __name__ == '__main__':
    D, L = load('Data/Train.txt')
    Dt, Lt = load('Data/Test.txt')

    DTR, LTR = randomize(D, L)
    DTE, LTE = randomize(Dt, Lt)
    #plot_features(DTR, LTR)
    validation(DTR, LTR)



