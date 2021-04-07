import pickle
import numpy as np


percentage = [0.1, 0.25, 0.5, 0.75, 0.9]


def mean(x):
    return np.array(x).mean()


def check_single_results(file_name):
    with open(file_name, 'rb') as input:
        x = pickle.load(input)

    # train_NN = [r['train_NN'].numpy() for r in x]
    test_NN = [r['test_NN'].numpy() for r in x]
    test_KENN = [r['test_KENN'].numpy() for r in x]
    # train_KENN = [r['train_KENN'].numpy() for r in x]

    print('NN mine: ' + str(mean(test_NN)))
    print('KENN: ' + str(mean(test_KENN)))
    return str(mean(test_KENN) - mean(test_NN))


def check_all_results(directory, results_NN_Marra, results_SBR, results_RNM):
    for i in range(5):
        p = percentage[i]
        r_NN_Marra = results_NN_Marra[i]
        r_SBR = results_SBR[i]
        r_RNM = results_RNM[i]

        print('   === Percentage of training: ' + str(int(p * 100)) + '% ===   ')
        print('Delta E2E: ' + check_single_results(directory + 'e2e/results_' + str(int(p * 100))))
        print('Delta GREEDY: ' + check_single_results(directory + 'greedy/results_' + str(int(p * 100))))
        print('NN Marra: ' + str(r_NN_Marra))
        print('Delta SBR: ' + str(r_SBR - r_NN_Marra))
        print('Delta RNM: ' + str(r_RNM - r_NN_Marra) + '\n')



# Inductive
results_NN_Marra_i = [0.645, 0.674, 0.707, 0.717, 0.723]
results_SBR_i = [0.650, 0.682, 0.712, 0.719, 0.726]
results_RNM_i = [0.685, 0.709, 0.726, 0.726, 0.732]

print('INDUCTIVE')
check_all_results('', results_NN_Marra_i, results_SBR_i, results_RNM_i)


# Transductive
# results_NN_Marra_t = [0.640, 0.667, 0.695, 0.708, 0.726]
# results_SBR_t = [0.703, 0.729, 0.747, 0.764, 0.780]
# results_RNM_t = [0.708, 0.735, 0.753, 0.766, 0.780]


# print('\n\nTRANSDUCTIVE')
# check_all_results('transductive/', results_NN_Marra_t, results_SBR_t, results_RNM_t)