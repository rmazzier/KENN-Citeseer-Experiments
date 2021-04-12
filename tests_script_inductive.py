import numpy as np
from pre_elab import generate_dataset
import training_standard as ts
import training_inductive as t
import training_greedy_inductive as tg
import pickle
import tensorflow as tf
import settings as s

def run_tests_inductive(
    n_runs, 
    include_greedy=True, 
    include_e2e=True, 
    save_results=True, 
    custom_training_dimensions=False, 
    verbose=True, 
    random_seed=s.RANDOM_SEED):


    # SET RANDOM SEED for tensorflow e numpy
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    training_dimensions = []
    if not custom_training_dimensions:
        print("No custom training dimensions found.")
        training_dimensions = [0.1, 0.25, 0.5, 0.75, 0.9]
        print("Using default training dimensions: {}".format(training_dimensions))
    else:
        training_dimensions = custom_training_dimensions

    results_e2e = {}
    results_greedy = {}

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
                print("--- Starting Base NN Training ---")
                results_e2e[td_string].setdefault('NN', []).append(ts.train_and_evaluate_standard(td, verbose=verbose)[3])
                print("--- Starting Inductive KENN Training ---")
                results_e2e[td_string].setdefault('KENN', []).append(t.train_and_evaluate_kenn_inductive(td, verbose=verbose))

            if include_greedy:
                print("--- Starting Base NN & Inductive Greedy KENN Training ---")
                res_nn, res_kenn = tg.train_and_evaluate_kenn_inductive_greedy(td, verbose=verbose)
                results_greedy[td_string].setdefault('NN', []).append(res_nn)
                results_greedy[td_string].setdefault('KENN', []).append(res_kenn)

        if save_results:
            if include_e2e:
                with open('./results/e2e/results_inductive(3layers)_{}runs'.format(n_runs), 'wb') as output:
                    pickle.dump(results_e2e, output)
            if include_greedy:
                with open('./results/greedy/results_inductive(3layers)_{}runs'.format(n_runs), 'wb') as output:
                    pickle.dump(results_greedy, output)
        
    return (results_e2e, results_greedy)

if __name__ == '__main__':
    run_tests_inductive(n_runs=3, custom_training_dimensions=[0.75, 0.90])