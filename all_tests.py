from tests_script_transductive import run_tests_transductive
from tests_script_inductive import run_tests_inductive

if  __name__ == '__main__':
    
    n_runs = 100

    run_tests_inductive(
        n_runs=n_runs,
        include_greedy=True,
        include_e2e=True,
        save_results=True,
        verbose=True)

    run_tests_transductive(
        n_runs=n_runs,
        include_greedy=True,
        include_e2e=True,
        save_results=True,
        verbose=True)
