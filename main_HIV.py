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
from Environment import HIVEnvironment, ModelEvaluate
from HyperParametersFind import ControlInput
from Kernel import HIVKernel
from Exploration import RandomExploration, TraceExploration

def reset_explore(kernel):
    true_instance = HIVEnvironment(A=A_true,dimu=dimu,noise_cov=noise_cov,states_init=states_init,kernel=kernel,dt=0.01)
    est_instance = copy.deepcopy(true_instance)
    est_instance.reset_dynamics()
    forcing = ControlInput(signal_type=prediction_parameters['signal_type'])
    evaluate = ModelEvaluate(true_instance, forcing, metrics, prediction_parameters=prediction_parameters, controller_parameters=controller_parameters)
    return true_instance, est_instance, evaluate

lambda1_sys = 1
d_sys = 0.1
beta_sys = 1
a_sys = 0.2
p1_sys = 1
p2_sys = 1
c1_sys = 0.03
c2_sys = 0.06
b1_sys = 0.1
b2_sys = 0.01
q_sys = 0.5
h_sys = 0.1 
eta_sys = 0.9799
states_init = torch.tensor([10, 0.1, 0.1, 0.1, 0.1])
A_true = torch.tensor([[lambda1_sys, -d_sys, 0.0, 0.0, 0.0, 0.0, 0.0, -beta_sys, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, beta_sys*eta_sys],
                       [0.0, 0.0, -a_sys, 0.0, 0.0, 0.0, 0.0, beta_sys, 0.0, -p1_sys, -p2_sys, 0.0, 0.0, 0.0, -beta_sys*eta_sys],
                       [0.0, 0.0, 0.0, -b2_sys, 0.0, 0.0, 0.0, 0.0, -c2_sys*q_sys, 0.0, 0.0, c2_sys, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, -b1_sys, 0.0, 0.0, 0.0, 0.0, c1_sys, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, -h_sys, 0.0, 0.0, c2_sys*q_sys, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
kernel = HIVKernel()
noise_cov = 1e-2
dimu = 1


N_explore = 25 # Number of Trajectories Considered in Each Step
N_horizon = 10 # Seconds to simulate
sim_frac = 1.0
epochs = [2,1,1,1,1,1,1,1,1]

experiment_repeat = 100
n_processes, n_processes_hyperparameters = 10, 25
signal_type = "ChirpLinear" # "SinWave" or "Sin2Wave" or "ChirpLinear" or "SchroederSweep" or "PRBS"
amplitude = 1
ExpObj = "E_Optimal" # "A_Optimal" or "D_Optimal" or "E_Optimal" or "G_Optimal"
random_seed = 0
metrics = ["normalized","mape","mae","mse","lq_cost"] # "default" or "normalized" or "mape" or "mae" or "mse" or "r2" or "lq_cost"

# Prediction Evaulation
prediction_parameters = {
    'states_init' : np.array([0.3, 5, 0.0, 0.2, 1.1]),
    'sim_time' : 10,
    'signal_type' : 'SchroederSweep',
    'hyperparams' : np.array([30, 22])
}

# Reference Tracking Evaulation
a, b, c, d, e = ca.SX.sym('a'), ca.SX.sym('b'), ca.SX.sym('c'), ca.SX.sym('d'), ca.SX.sym('e') # States variables
u = ca.SX.sym('u') # Control variables
controller_parameters = {
    'states': ca.vertcat(a, b, c, d, e),
    'controls': ca.vertcat(u),
    'states_mapped': [1, a, b, c, d, e, u, a*b, b*c, b*d, b*e, a*b*c, a*b*d, a*b*e, a*b*u],
    'N' : 10,
    'Q': ca.diagcat(3e5, 3e5, 3e5, 3e5, 3e5),
    'R': ca.diagcat(1),
    'x_lb': [-10000, -10000, -10000, -10000, -10000],
    'x_ub': [10000, 10000, 10000, 10000, 10000],
    'u_lb': [-10000],
    'u_ub': [10000],
    'states_init': [0.3, 5, 0.0, 0.2, 1.1],
    'states_target': [0.291262135922330, 3.33333333333333, 0, 0.0912621359223301, 0],
    'dt' : 2e-2,
    'dt_simulate' : 2e-2,
    'sim_time' : 100

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