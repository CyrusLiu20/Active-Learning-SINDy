import numpy as np
import time
import pysindy as ps
import torch
import casadi as ca

import matplotlib.pyplot as plt
import copy
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

# User Defined
from Environment import F8Environment, ModelEvaluate
from HyperParametersFind import ControlInput
from Kernel import F8Kernel, SymbolicPolynomialLibrary
from Exploration import RandomExploration, TraceExploration

def reset_explore(kernel):
    true_instance = F8Environment(A=A_true,dimu=dimu,noise_cov=noise_cov,states_init=states_init,kernel=kernel,dt=0.01)
    est_instance = copy.deepcopy(true_instance)
    est_instance.reset_dynamics()
    forcing = ControlInput(signal_type=prediction_parameters['signal_type'], amplitude=amplitude)
    evaluate = ModelEvaluate(true_instance, forcing, metrics, prediction_parameters=prediction_parameters, controller_parameters=controller_parameters)
    return true_instance, est_instance, evaluate

states_init = torch.tensor([0.3, 0.1, -0.5])
A_true = torch.tensor([[0,-0.877,0,1,-0.215,0.47,0,-0.088,0,-0.019,0,0,0,0,0,3.846,0,-1,0.28,0,0,0,0,0,0.47,0,0,0,0,0,0,0,0,0,0.63],
                       [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                       [0,-4.208,0,-0.396,-20.967,-0.47,0,0,0,0,0,0,0,0,0,-3.564,0,0,6.265,0,0,0,0,0,46,0,0,0,0,0,0,0,0,0,61.4]])
degree = 3
kernel = F8Kernel(degree=degree)
noise_cov = 1e-2
dimu = 1


N_explore = 25 # Number of Trajectories Considered in Each Step
N_horizon = 10 # Seconds to simulate
sim_frac = 1
# epochs = [2,1,1,1,1,1,1,1,1]
epochs = [2,1]

experiment_repeat = 1
n_processes, n_processes_hyperparameters = 1, 1
signal_type = "ChirpLinear" # "SinWave" or "Sin2Wave" or "ChirpLinear" or "SchroederSweep" or "PRBS"
amplitude = 0.01
ExpObj = "A_Optimal" # "A_Optimal" or "D_Optimal" or "E_Optimal" or "G_Optimal"
random_seed = 0
metrics = ["normalized","mape","mae","mse"] # "default" or "normalized" or "mape" or "mae" or "mse" or "r2" or "lq_cost"


# Prediction Evaulation
prediction_parameters = {
    'states_init' : np.array([0.1, 0, 0]),
    'sim_time' : 10,
    'signal_type' : 'SinWave',
    'hyperparams' : np.array([0.3])
}

# Reference Tracking Evaulation
a, b, c = ca.SX.sym('a'), ca.SX.sym('b'), ca.SX.sym('c') # States variables
u = ca.SX.sym('u') # Control variables
states_concat = [a,b,c,u]   
library = SymbolicPolynomialLibrary()
library.fit(variables=states_concat,max_degree=degree,include_bias=True)
states_mapped = library.get_feature_map()
controller_parameters = {
    'states': ca.vertcat(a, b, c),
    'controls': ca.vertcat(u),
    'states_mapped': states_mapped,
    'N' : 5,
    'Q': ca.diagcat(1e3, 1e2, 1e0),
    'R': ca.diagcat(1),
    'x_lb': [-10000, -10000, -10000],
    'x_ub': [10000, 10000, 10000],
    'u_lb': [-10000],
    'u_ub': [10000],
    'states_init': [0.1, 0, 0],
    'states_target': [-0.1, 0, 0],
    'dt' : 1e-2,
    'dt_simulate' : 1e-2,
    'sim_time' : 10

}



def run_experiment(experiment_index, random_seed=None):
    print("Experiment Index : ", experiment_index)
    X_history_all, U_history_all, Hyperparams_history_all, error_history_all = [],[],[],[]
    np.random.seed(random_seed)

    # Random Exploration
    print("Random Exploration : ", experiment_index)
    true_instance, est_instance, evaluate = reset_explore(kernel=kernel)
    explore_policy_rand = RandomExploration(N_explore, N_horizon, signal_type, amplitude, evaluate, kernel, metrics=metrics)
    est_instance, cov, U_history, X_history, Hyperparams_history, error_history = explore_policy_rand.system_id(
        true_instance, est_instance, epochs, closed_form=False, verbose=1)

    X_history_all.append(X_history)
    U_history_all.append(U_history)
    Hyperparams_history_all.append(Hyperparams_history)
    error_history_all.append(error_history)

    # Optimal Exploration
    print("Optimal Exploration : ", experiment_index)
    true_instance, est_instance, evaluate = reset_explore(kernel=kernel)
    explore_policy_trace = TraceExploration(N_explore, N_horizon, n_processes_hyperparameters, signal_type, amplitude, evaluate, kernel, sim_frac=sim_frac, ExpObj=ExpObj, metrics=metrics)
    est_instance, cov, U_history, X_history, Hyperparams_history, error_history = explore_policy_trace.system_id(
        true_instance, est_instance, epochs, hyperparams_control=Hyperparams_history, closed_form=False, verbose=1)

    X_history_all.append(X_history)
    U_history_all.append(U_history)
    Hyperparams_history_all.append(Hyperparams_history)
    error_history_all.append(error_history)

    return X_history_all, U_history_all, Hyperparams_history_all, error_history_all


def run_experiment_wrapper(experiment_index):
    # try:
    X_history_all, U_history_all,  Hyperparams_history_all, error_history_all = run_experiment(experiment_index,random_seed=experiment_index+random_seed)
    return experiment_index, X_history_all, U_history_all, Hyperparams_history_all, error_history_all
    # except Exception as e:
    #     print(f"Experiment {experiment_index} failed with error: {e}")
    #     return experiment_index, None, None, None, None


if __name__ == "__main__":
    print("Experiment has Started")
    start_time = time.time()
    X_history_all, U_history_all, error_history_all, Hyperparams_history_all = [None]*(experiment_repeat), [None]*(experiment_repeat), [None]*(experiment_repeat), [None]*(experiment_repeat)

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = list(executor.map(run_experiment_wrapper, range(experiment_repeat)))

        # futures = [executor.submit(run_experiment_wrapper, i) for i in range(experiment_repeat)]
        # completed_futures, _ = wait(futures, return_when=ALL_COMPLETED)
        # results = [future.result() for future in completed_futures]

    for experiment_index, X_history, U_history, Hyperparams_history, error_history in results:
        if X_history is not None:
            X_history_all[experiment_index] = X_history
            U_history_all[experiment_index] = U_history
            Hyperparams_history_all[experiment_index] = Hyperparams_history
            error_history_all[experiment_index] = error_history


    end_time = time.time()
    elapsed_time = end_time - start_time
    torch.save(signal_type, 'SimulationData//signal_type.pt')
    torch.save(metrics, 'SimulationData//metrics.pt')
    torch.save(ExpObj, 'SimulationData//ExpObj.pt')
    torch.save(U_history_all, 'SimulationData//U_history_all.pt')
    torch.save(X_history_all, 'SimulationData//X_history_all.pt')
    torch.save(Hyperparams_history_all, 'SimulationData//Hyperparams_history_all.pt')
    torch.save(error_history_all, 'SimulationData//error_history_all.pt')
    torch.save(torch.tensor(epochs), 'SimulationData//epochs.pt')
    print("Experiment has Finished")
    print("Time spent for simulation : ", elapsed_time, "seconds")