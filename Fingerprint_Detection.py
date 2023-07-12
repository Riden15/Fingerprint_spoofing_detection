from Evaluation.Comparison.compare_GMMvsSVM_RBF import compare_evaluation_GMM_vs_SVM_RBF
from Evaluation.Comparison.compare_GMMvsSVM_RBF_calibrated import compare_evaluation_GMM_vs_SVM_RBF_calibrated
from Evaluation.Comparison.compare_MVGvsGMM import compare_evaluation_MVG_vs_GMM
from Evaluation.Comparison.compare_MVGvsSVM_RBF import compare_evaluation_MVG_vs_SVM_RBF
from Evaluation.Comparison.compare_MVGvsSVM_RBF_calibrated import compare_evaluation_MVG_vs_SVM_RBF_calibrated
from Evaluation.Comparison.score_calibration_SVM_RBF import evaluation_SVM_RBF_score_calibration
from Evaluation.evaluation_GMM import evaluation_GMM
from Evaluation.evaluation_LR import evaluation_LR
from Evaluation.evaluation_MVG import evaluation_MVG
from Evaluation.evaluation_SVM import evaluation_SVM
from Evaluation.evaluation_SVM_RBF import evaluation_SVM_RBF
from Evaluation.evaluation_SVM_polynomial import evaluation_SVM_polynomial
from Utility_functions.plot_features import plot_features
from Validation.Comparison.compare_GMMvsSVM_RBF import compare_validation_GMM_vs_SVM_RBF
from Validation.Comparison.compare_GMMvsSVM_RBF_calibrated import compare_validation_GMM_vs_SVM_RBF_calibrated
from Validation.Comparison.compare_MVGvsGMM import compare_validation_MVG_vs_GMM
from Validation.Comparison.compare_MVGvsSVM_RBF_calibrated import compare_validation_MVG_vs_SVM_RBF_calibrated
from Validation.Comparison.score_calibration_SVM_RBF import SVM_RBF_score_calibration
from Validation.validation_GMM import validation_GMM
from Validation.validation_LR import validation_LR
from Validation.validation_LR_quad import validation_LR_quad
from Validation.validation_MVG import validation_MVG
from Validation.validation_SVM import validation_SVM
from Validation.validation_SVM_RBF import validation_SVM_RBF
from Validation.validation_SVM_polynomial import *
from Validation.Comparison.compare_MVGvsSVM_RBF import compare_validation_MVG_vs_SVM_RBF


def validation(DTR, LTR):
    print("############    MVG    ##############")
    validation_MVG(DTR, LTR, 5)  # FINITO

    print("###########      LR      ##############")
    L = [0.4]  # da provare anche per LR normale, 0.4 è l'ottimale per il quadratic
    validation_LR(DTR, LTR, L, 5)  # FINITO

    print("############    Quadratic Logistic Regression    ##############")
    validation_LR_quad(DTR, LTR, L, 5)  # DA TOGLIERE, NON ANDAVA FATTO

    print("############    Support Vector Machine - Linear    ##############")
    K_arr = [0.1, 1.0, 10.0]
    C_arr = [0.01, 0.1, 1.0, 10.0]
    validation_SVM(DTR, LTR, K_arr, C_arr, 5)  # FINITO

    print("############    Support Vector Machine - Quadratic    ##############")
    K_arr = [0.1, 1, 10]
    C_arr = [0.01, 0.1, 1.0, 10]
    constant = [0, 1]
    validation_SVM_polynomial(DTR, LTR, K_arr, C_arr, constant, 5)

    print("############    Support Vector Machine - RBF    ##############")
    K_arr = [0.1, 1, 10]
    C_arr = [1, 10, 100]
    gamma_arr = [0.001, 0.0001]
    validation_SVM_RBF(DTR, LTR, K_arr, gamma_arr, C_arr, 5)

    print("############    Gaussian Mixture Models   ##############")
    validation_GMM(DTR, LTR, 5)

    compare_validation_MVG_vs_SVM_RBF(DTR, LTR, 5)
    compare_validation_MVG_vs_GMM(DTR, LTR, 5)
    compare_validation_GMM_vs_SVM_RBF(DTR, LTR, 5)
    SVM_RBF_score_calibration(DTR, LTR, 5)
    compare_validation_MVG_vs_SVM_RBF_calibrated(DTR, LTR, 5)
    compare_validation_GMM_vs_SVM_RBF_calibrated(DTR, LTR, 5)


def evaluation(DTR, LTR, DTE, LTE):
    print("############    MVG    ##############")
    evaluation_MVG(DTR, LTR, DTE, LTE)  # FINITO

    print("###########      LR      ##############")
    L = 0.4
    evaluation_LR(DTR, LTR, DTE, LTE, L)  # FINITO

    print("############    Support Vector Machine - Linear    ##############")
    K = 1.0
    C = 1.0
    evaluation_SVM(DTR, LTR, DTE, LTE, K, C)  # FINITO

    print("############    Support Vector Machine - Quadratic    ##############")
    K = 1
    C = 0.1
    constant = 1
    degree = 2
    evaluation_SVM_polynomial(DTR, LTR, DTE, LTE, K, C, constant, degree)

    print("############    Support Vector Machine - RBF    ##############")
    K = 0.1
    C = 10
    gamma = 0.001
    evaluation_SVM_RBF(DTR, LTR, DTE, LTE, K, gamma, C)

    print("############    Gaussian Mixture Models   ##############")
    comp = 3
    evaluation_GMM(DTR, LTR, DTE, LTE, comp)

    compare_evaluation_MVG_vs_SVM_RBF(DTR, LTR, DTE, LTE)
    compare_evaluation_MVG_vs_GMM(DTR, LTR, DTE, LTE)
    compare_evaluation_GMM_vs_SVM_RBF(DTR, LTR, DTE, LTE)
    evaluation_SVM_RBF_score_calibration(DTR, LTR, DTE, LTE)
    compare_evaluation_MVG_vs_SVM_RBF_calibrated(DTR, LTR, DTE, LTE)
    compare_evaluation_GMM_vs_SVM_RBF_calibrated(DTR, LTR, DTE, LTE)


if __name__ == '__main__':
    D, L = load('Data/Train.txt')
    Dt, Lt = load('Data/Test.txt')

    DTR, LTR = randomize(D, L)
    DTE, LTE = randomize(Dt, Lt)
    plot_features(DTR, LTR)
    validation(DTR, LTR)
    evaluation(DTR, LTR, DTE, LTE)
