import torch
import numpy as np
from functools import partial
from scipy.integrate import solve_ivp
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from Kernel import HIVKernel
from Environment import Environment
import multiprocessing
lock = multiprocessing.Lock()

class hyperparameters_find:
    
    def __init__(self,H,n_processes,sim_frac,signal_type,amplitude,ExpObj,kernel):
        self.H = H
        self.n_processes = n_processes
        self.sim_frac = sim_frac
        self.signal_type = signal_type
        self.amplitude = amplitude
        self.ExpObj = ExpObj

        self.integrator_keywords = {}
        self.integrator_keywords["rtol"] = 1e-8
        self.integrator_keywords["method"] = "LSODA"
        self.integrator_keywords["atol"] = 1e-8
        self.kernel = kernel

        self.forcing = ControlInput(self.signal_type, self.amplitude)
        self.optimality = OptimalityCriteria(self.ExpObj)
        self.threshold = 1e3

    def random_parameters(self, n_explore):
        
        if self.signal_type=="SinWave":
            # frequencies = np.random.uniform(0.5, 15, size=n_explore)
            frequencies = np.random.uniform(0.01, 25, size=n_explore)
            hyperparams = frequencies.reshape(-1,1)
        elif self.signal_type=="Sin2Wave":
            frequencies1 = np.random.uniform(0.01, 25, size=n_explore)
            frequencies2 = np.random.uniform(0.01, 5, size=n_explore)
            hyperparams = np.column_stack((frequencies1, frequencies2))
        elif self.signal_type=="ChirpLinear":
            f1 = np.random.uniform(0.02,1, size=n_explore)
            t1 = np.random.uniform(1,10, size=n_explore)
            hyperparams = np.column_stack((f1, t1))
        elif self.signal_type=="SchroederSweep":
            period = np.random.uniform(20,100, size=n_explore)
            n_harmonics = np.random.randint(10, 30+1, size=n_explore)
            hyperparams = np.column_stack((period, n_harmonics))
        elif self.signal_type=="PRBS":
            random_seeds = np.random.randint(0, 1e6, size=n_explore)
            interval_max = np.random.uniform(0.6, 2, size=n_explore)
            hyperparams = np.column_stack((random_seeds, interval_max))
        else:
            raise ValueError("Unrecognized Signal Type: {}".format(self.signal_type))

        return hyperparams

    def hyperparameters_find(self, est_instance, n_explore, n_data):

        hyperparams = self.random_parameters(n_explore)
        costs = torch.zeros(n_explore)

        t_explore = torch.arange(0,self.H*self.sim_frac,est_instance.dt)
        t_span = (t_explore[0], t_explore[-1])
        time_arg = (t_explore, t_span)
        est_instance_arg = est_instance.get_all_params()
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            results = list(executor.map(self.cost_find, [est_instance_arg] * n_explore, [time_arg] * n_explore, hyperparams, range(n_explore)))
            # futures = [executor.submit(self.cost_find, est_instance_arg, time_arg, hyperparams[i], i) for i in range(n_explore)]
            # completed_futures, _ = wait(futures, return_when=ALL_COMPLETED)
            # # Retrieve results
            # results = [future.result() for future in completed_futures]

        # Update costs based on the results
        for idx, cost in results:
            costs[idx] = cost

        # print(f"costs : {costs}")
        if self.ExpObj == "A_Optimal":
            # FW approximation for minimising tr(inv(cov))
            idx = torch.argmax(costs)
        elif self.ExpObj == "E_Optimal":
            # Maximising the minimum eigenvalue
            idx = torch.argmax(costs)
        elif self.ExpObj == "D_Optimal":
            # Minimising the log determinant of covariance/ Maximising the log determinant
            idx = torch.argmin(costs)
        else:
            raise ValueError("Unrecognized Optimality Criteria: {}".format(self.ExpObj))

        hyperparams_chosen = np.array([hyperparams[idx]])
        print("Hyperparameters Chosen : ", hyperparams_chosen)

        return hyperparams_chosen

    def cost_find(self, instance_arg, time_arg, hyperparams, idx):
        with lock:
            A,dimu,noise_cov,states_init,dt = instance_arg
            est_instance_parallel = Environment(A=A,dimu=dimu,noise_cov=noise_cov,states_init=states_init,kernel=self.kernel,dt=dt)

            t_explore, t_span = time_arg

            u_hist = self.forcing.control_data(hyperparams,t_explore,dimu)
            u_func = self.forcing.control_function(hyperparams=hyperparams)
            u_func_partial = partial(u_func)

            if self.signal_type!="PRBS":
                sol = solve_ivp(est_instance_parallel.dynamics_deri, t_span, states_init, events=[self.terminate1,self.terminate2],
                                        t_eval=t_explore,**self.integrator_keywords,args=(u_func_partial,))
            else:
                u_func_partial = partial(self.forcing.u_func_prbs)
                sol = solve_ivp(est_instance_parallel.dynamics_deri, t_span, states_init, events=[self.terminate1,self.terminate2],
                                        t_eval=t_explore,**self.integrator_keywords,args=(u_func_partial,))
                u_hist = np.column_stack([u_hist])

            x_hist = torch.tensor(sol.y.T)
            if sol.status==0:
                cost = self.optimality.cost(x_hist,u_hist,est_instance_parallel,ignore_frac=0)
            else:
                cost = float("NaN")
                print(f"Cost Find: {idx}, Integration Status: {sol.status}, Integration Message: {sol.message}")

        return idx, cost
    
    def terminate1(self, t, x, u_func):
        return np.array(x[0])-self.threshold
    def terminate2(self, t, x, u_func):
        return np.array(x[0])+self.threshold

class ControlInput:
    def __init__(self, signal_type, amplitude=1):
        self.signal_type = signal_type
        self.amplitude = amplitude

    def control_function(self, hyperparams):
        # One set of hyperparameters only (1D)
        if self.signal_type=="SinWave":
            def u_func(t):
                # sin(frequency*t)
                return self.amplitude*np.column_stack([np.sin(hyperparams[0]*t)])
        elif self.signal_type=="Sin2Wave":
            def u_func(t):
                # sin(frequency1*t) + sin(frequency2*t)^2
                u = np.sin(hyperparams[0]*t) + np.sin(hyperparams[1]*t)**2
                return self.amplitude*np.column_stack([u])
        elif self.signal_type=="ChirpLinear":
            # require debugging
            f0 = 0
            f1 = hyperparams[0]
            t1 = hyperparams[1]
            phi = 0
            beta = (f1 - f0) * (1 / t1)
            def u_func(t):
                u = np.cos(2 * np.pi * (beta / 2 * (t ** 2) + f0 * t + np.radians(phi) / 360))
                return self.amplitude*np.column_stack([u])
        elif self.signal_type=="SchroederSweep":
            Pf = hyperparams[0]
            K = int(hyperparams[1])
            def u_func(t):
                u = np.zeros_like(t)
                for i in range(1, K+1):
                    theta = 2 * np.pi / K * np.sum(np.arange(1, i+1))
                    u = u + np.array(np.sqrt(2/K) * np.cos(2 * np.pi * i * t / Pf + theta))
                return self.amplitude*np.column_stack([u])
        elif self.signal_type=="PRBS":
            def u_func(t):
                # index = np.searchsorted(self.t_prbs, t, side='right') - 1
                index = int(np.floor(t/self.dt))
                return self.u_prbs[index]
            
        return u_func
    
    def control_data(self, hyperparams, t_explore, dimu):
        # One set of hyperparameters only (1D)
        if self.signal_type=="SinWave":
            u_hist = self.amplitude*torch.sin(hyperparams[0]*t_explore).reshape(len(t_explore),dimu)
        elif self.signal_type=="Sin2Wave":
            u = torch.sin(hyperparams[0]*t_explore) + torch.sin(hyperparams[1]*t_explore)**2
            u_hist = self.amplitude*u.reshape(len(t_explore),dimu)
        elif self.signal_type=="ChirpLinear":
            # f0 = 0, phi = 0
            f1 = hyperparams[0]
            t1 = hyperparams[1]
            u_hist = self.amplitude*torch.cos(2 * np.pi * ((f1 / t1) / 2 * (t_explore ** 2))).reshape(len(t_explore),dimu)
        elif self.signal_type=="SchroederSweep":
            u_func = self.control_function(hyperparams=hyperparams)
            u_hist = u_func(t_explore)
        elif self.signal_type=="PRBS":
            random_seed = int(hyperparams[0])
            np.random.seed(random_seed)
            taulim = [0.1, hyperparams[1]]
            states = [0, 0.25, 0.5, 0.75 ,1]
            Nswitch = int(t_explore[-1]/taulim[0])
            Toffset = 0.0
            u_hist = self.prbs(taulim, Nswitch, states, t_explore, Toffset)
            self.u_prbs = u_hist
            self.t_prbs = t_explore
            self.dt = t_explore[1]-t_explore[0]

        return u_hist       

    def prbs(self, taulim, Nswitch, states, t, Toffset):
        # Generate random time intervals between switches
        tau_list = np.random.uniform(taulim[0], taulim[1], Nswitch)
        tau_list = np.insert(tau_list, 0, 0)
        tau_list = np.cumsum(tau_list) + Toffset

        # Randomly select values for the signal at each switch
        ub = np.random.choice(states, Nswitch)
        u = np.zeros_like(t)
        for i, tau in enumerate(tau_list[:-1]):
            mask = (t >= tau_list[i]) & (t < tau_list[i+1])
            u[mask] = ub[i]
        
        return u
    
    def u_func_prbs(self, t):
        # index = np.searchsorted(self.t_prbs, t, side='right') - 1
        index = int(np.floor(t/self.dt))
        u = self.u_prbs[index]
        return np.array([u])
    
class OptimalityCriteria:

    def __init__(self,optimality_criteria):
        self.optimality_criteria = optimality_criteria

    def cost(self,x_horizon,u_horizon,est_instance,ignore_frac=0):
        if self.optimality_criteria == "A_Optimal":           
            cost = self.A_optimal_cost(x_horizon,u_horizon,est_instance,ignore_frac=0)
        elif self.optimality_criteria == "E_Optimal":
            cost =  self.E_optimal_cost(x_horizon,u_horizon,est_instance,ignore_frac=0)
        elif self.optimality_criteria == "D_Optimal":
            cost =  self.D_optimal_cost(x_horizon,u_horizon,est_instance,ignore_frac=0)
        elif self.optimality_criteria == "G_Optimal":
            cost =  self.G_optimal_cost(x_horizon,u_horizon,est_instance,ignore_frac=0)
        else:
            raise ValueError("Unrecognized Optimality Criteria: {}".format(self.optimality_criteria))
        return cost

    def A_optimal_cost(self,x_horizon,u_horizon,est_instance,ignore_frac=0):
        
        n_data = len(x_horizon)
        covs = torch.zeros_like(est_instance.get_cov(x_horizon[0],u_horizon[0]))
        for i in range(int(ignore_frac*n_data),n_data):
            covs = covs + est_instance.get_cov(x_horizon[i],u_horizon[i])
        
        R = self.hess_to_cost(covs,est_instance.dimx)
        cost = torch.trace(R.type(torch.FloatTensor) @ covs.type(torch.FloatTensor))/(n_data)
        return cost
    
    def hess_to_cost(self,cov,dimx):
        # compute quadratic cost to use at step n of online FW procedure
        dimz = cov.shape[0]
        R = torch.zeros(dimz,dimz)
        cov_inv = torch.linalg.inv(cov + 0.0001*torch.eye(cov.shape[0]))
        for i in range(dimz):
            for j in range(dimz):
                eij = torch.zeros(dimz,dimz)
                eij[i,j] = 1
                temp = torch.kron(torch.eye(dimx).type(torch.DoubleTensor), cov_inv.type(torch.DoubleTensor) @ eij.type(torch.DoubleTensor) @ cov_inv.type(torch.DoubleTensor))
                R[i,j] = torch.trace(temp)
        e,_ = torch.linalg.eig(R)

        if torch.min(torch.real(e)) < 0:
            R = R.T @ R
            U,S,V = torch.svd(R)
            R = U @ torch.diag(torch.sqrt(S)) @ U.T

        R = R / torch.max(R)
        R = R / torch.linalg.matrix_norm(R,2)
        return R.detach()
    
    def D_optimal_cost(self,x_horizon,u_horizon,est_instance,ignore_frac=0):
        n_data = len(x_horizon)
        if n_data != len(u_horizon):
            print(f"Warning: Number of data points to be considered: {n_data} instead of {len(u_horizon)}")
        covs = torch.zeros_like(est_instance.get_cov(x_horizon[0],u_horizon[0]))
        for i in range(int(ignore_frac*n_data),n_data):
            covs = covs + est_instance.get_cov(x_horizon[i],u_horizon[i])

        cost = torch.log(torch.linalg.det(covs))/(n_data)
        return cost
    
    def E_optimal_cost(self,x_horizon,u_horizon,est_instance,ignore_frac=0):
        n_data = len(x_horizon)
        covs = torch.zeros_like(est_instance.get_cov(x_horizon[0],u_horizon[0]))
        for i in range(int(ignore_frac*n_data),n_data):
            covs = covs + est_instance.get_cov(x_horizon[i],u_horizon[i])

        fim = torch.inverse(covs + 0.001 * torch.eye(covs.shape[0]))
        e,_ = torch.linalg.eig(fim)
        cost = torch.min(torch.real(e))/(n_data)
        return cost
    
    def G_optimal_cost(self,x_horizon,u_horizon,est_instance,ignore_frac=0):
        n_data = len(x_horizon)
        if n_data != len(u_horizon):
            print(f"Warning: Number of data points to be considered: {n_data} instead of {len(u_horizon)}")
        hats = torch.zeros_like(est_instance.get_cov(x_horizon[0],u_horizon[0]))
        for i in range(int(ignore_frac*n_data),n_data):
            hats = hats + est_instance.get_hat(x_horizon[i],u_horizon[i])

        cost = torch.max(torch.diag(hats))/(n_data)
        return cost