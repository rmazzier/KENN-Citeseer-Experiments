from tests_script_transductive import run_tests_transductive
from tests_script_inductive import run_tests_inductive
import argparse
import os


def parsing():
    parser = argparse.ArgumentParser(
        description="Run tests for both inductive and transductive paradigm")
    parser.add_argument(
        '-n',
        type=int,
        default=500,
        help='The total number of runs for each learning paradigm')
    parser.add_argument(
        '-ind',
        action='store_true',
        help='To perform the inductive training;')
    parser.add_argument(
        '-tra',
        action='store_true',
        help='To perform the transductive training;')
    parser.add_argument(
        '-gr',
        action='store_true',
        help='To perform the greedy training;')
    parser.add_argument(
        '-e2e',
        action='store_true',
        help='To perform the end to end training;')

    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':

    directory = "results/e2e"

    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = "results/greedy"  #TODO: remove greedy

    if not os.path.exists(directory):
        os.makedirs(directory)

    args = parsing()
    n_runs = args['n']

    for n_layers in range(1, 7):
        if (args['ind']):
            run_tests_inductive(
                n_runs=n_runs,
                n_layers=n_layers,
                include_greedy=args['gr'],
                include_e2e=args['e2e'],
                save_results=True,
                verbose=True)

        if (args['tra']):
            run_tests_transductive(
                n_runs=n_runs,
                n_layers=n_layers,
                include_greedy=args['gr'],
                include_e2e=args['e2e'],
                save_results=True,
                verbose=True)
