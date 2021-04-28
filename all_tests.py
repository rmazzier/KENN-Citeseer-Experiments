from tests_script_transductive import run_tests_transductive
from tests_script_inductive import run_tests_inductive
import argparse

def parsing():
    parser = argparse.ArgumentParser(description="Run tests for both inductive and transductive paradigm")
    parser.add_argument(
        '-n',
        type=int,
        default=100,
        help='The total number of runs for each learning paradigm (default=True)')
    parser.add_argument(
        '-ind',
        type=bool,
        default=True,
        help='Whether to perform or not the inductive training (default=True)')
    parser.add_argument(
        '-tra',
        type=bool,
        default=True,
        help='Whether to perform or not the transductive training (default=True)')
    parser.add_argument(
        '-gr',
        type=bool,
        default=True,
        help='Whether to perform or not the greedy training (default=True)')
    parser.add_argument(
        '-e2e',
        type=bool,
        default=True,
        help='Whether to perform or not the end to end training (default=True)')

    args = vars(parser.parse_args())
    return args

if  __name__ == '__main__':
    
    args = parsing()
    n_runs = args['n']

    if (args['ind']):
        run_tests_inductive(
            n_runs=n_runs,
            include_greedy=args['gr'],
            include_e2e=args['e2e'],
            save_results=True,
            verbose=True)

    if (args['tra']):
        run_tests_transductive(
            n_runs=n_runs,
            include_greedy=args['gr'],
            include_e2e=args['e2e'],
            save_results=True,
            verbose=True)
