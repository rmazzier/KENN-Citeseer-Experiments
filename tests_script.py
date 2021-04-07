import numpy as np
from pre_elab import generate_dataset
import training as t
import training_greedy as tg
import pickle

def mean(x):
    return np.array(x).mean()

def run_tests_inductive(n_runs, include_greedy=True, include_e2e=True, save_results=True, custom_training_dimensions=False):
    training_dimensions = []
    if not custom_training_dimensions:
        print("No custom training dimensions found.")
        training_dimensions = [0.1, 0.25, 0.5, 0.75, 0.9]
        print("Using default training dimensions: {}".format(training_dimensions))
    else:
        training_dimensions = custom_training_dimensions

    results_e2e = {}
    results_greedy = {}
    #generate_dataset(td)

    for td in training_dimensions:
        td_string = str(int(td * 100))
        print(' ========= Start training (' + td_string + '%)  =========')
        results_e2e.setdefault(td_string, {})
        results_greedy.setdefault(td_string, {})

        # results e2e will be dictionaries like this:
        # {'10' : {'NN': [list of n_runs dictionaries containing all the stats], 'KENN': [same a before]},
        #  '25' : {'NN': [list of n_runs dictionaries containing all the stats], 'KENN': [same a before]}, 
        #   ...}
        
        for i in range(n_runs):
            print('Generate new dataset: iteration number ' + str(i))
            # spostare il generate_dataset(td) fuori dal for per non avere un dataset diverso a ogni try
            # provare!
            generate_dataset(td, verbose=False)

            if include_e2e:
                results_e2e[td_string].setdefault('NN', []).append(t.train_and_evaluate_standard_inductive(td))
                results_e2e[td_string].setdefault('KENN', []).append(t.train_and_evaluate_kenn_inductive(td))

            if include_greedy:
                results_greedy[td_string].setdefault('NN', []).append(tg.train_and_evaluate_kenn_inductive_greedy(td)[0])
                results_greedy[td_string].setdefault('KENN', []).append(tg.train_and_evaluate_kenn_inductive_greedy(td)[1])

        if save_results:
            if include_e2e:
                with open('./e2e/results_inductive(3layers)_{}runs'.format(n_runs), 'wb') as output:
                    pickle.dump(results_e2e, output)
            if include_greedy:
                with open('./greedy/results_inductive(3layers)_{}runs'.format(n_runs), 'wb') as output:
                    pickle.dump(results_greedy, output)


if __name__ == '__main__':
    run_tests_inductive(n_runs=20)