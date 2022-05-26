import numpy as np
from pre_elab import generate_dataset
import training_standard as ts
import training_transductive as t
import training_greedy_transductive as tg
import pickle
import tensorflow as tf
import settings as s
import time


def run_tests_transductive(
        n_runs,
        n_layers,
        include_greedy=True,
        include_e2e=True,
        save_results=True,
        custom_training_dimensions=False,
        verbose=True,
        random_seed=s.RANDOM_SEED):

    # SET RANDOM SEED for tensorflow and numpy
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
            generate_dataset(td, verbose=False)

            if include_e2e:
                print("--- Starting Base NN Training ---")
                start_time = time.time()
                results = ts.train_and_evaluate_standard(td, verbose=verbose)[3]
                end_time = time.time()
                results['time'] = end_time - start_time
                results_e2e[td_string].setdefault('NN', []).append(results)

                print("--- Starting KENN Transductive Training ---")
                start_time = time.time()
                r = t.train_and_evaluate_kenn_transductive(td, n_layers, verbose=verbose)
                end_time = time.time()

                r['time'] = end_time - start_time
                results_e2e[td_string].setdefault('KENN', []).append(r)

            if include_greedy:
                print("--- Starting KENN Greedy Training ---")
                res_nn, res_kenn = tg.train_and_evaluate_kenn_transductive_greedy(
                    td, verbose=verbose)
                results_greedy[td_string].setdefault('NN', []).append(res_nn)
                results_greedy[td_string].setdefault(
                    'KENN', []).append(res_kenn)

        if save_results:
            if include_e2e:
                with open('./results/e2e/results_transductive_{}runs_{}layers'.format(n_runs, n_layers), 'wb') as output:
                    pickle.dump(results_e2e, output)
            if include_greedy:
                with open('./results/greedy/results_transductive_{}runs'.format(n_runs), 'wb') as output:
                    pickle.dump(results_greedy, output)

    return (results_e2e, results_greedy)


if __name__ == "__main__":
    run_tests_transductive(3, include_greedy=True, include_e2e=True,
                           save_results=False, custom_training_dimensions=[0.90])
