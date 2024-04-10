import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial

from HyperParametersFind import hyperparameters_find, ControlInput

class RandomExploration:
    
    def __init__(self,N,H,signal_type,amplitude,evaluate,kernel,metrics="normalized"):

        self.N = N
        self.H = H
        self.signal_type = signal_type
        self.amplitude = amplitude
        self.evaluate = evaluate
        self.kernel = kernel
        self.metrics = metrics

        self.optimal_params = hyperparameters_find(self.H,None,None,self.signal_type,self.amplitude,None,self.kernel)
        self.forcing = ControlInput(self.signal_type, self.amplitude)
    
    def system_id(self,true_instance,est_instance,epochs,closed_form=True,verbose=1):
        x = est_instance.get_states_init()
        U_history, X_history, Frequencies_history, error_history = [], [], [], []
        cov = None
        
        for epoch_idx, i in enumerate(range(len(epochs))):     
            x_history, u_history, frequencies_history,  input_power, est_hessian, cov = self.explore(true_instance,est_instance,epochs[epoch_idx],epoch_idx,past_cov=cov)
            
            U_history = U_history + u_history
            X_history = X_history + x_history
            Frequencies_history = Frequencies_history + frequencies_history
            
           
            est_instance.update_estimates(X_history,U_history,closed_form=closed_form)
            
            # error_history.append(true_instance.compute_est_error(est_instance.get_dynamics(), metric=self.metrics))
            error_history.append(self.evaluate.compute_est_error(est_instance))
            model = est_instance.get_dynamics().numpy()
            if verbose == 0 or verbose == 1:
                print("Epoch ",str(epoch_idx)," :  loss :",error_history[epoch_idx])
                # print("Epoch", str(epoch_idx)," : loss :",true_instance.compute_est_error(est_instance.get_dynamics(), metrics="default"))
            if verbose == 0:
                print("Model Dynamics :\n",np.round(model,3))   
        
        return est_instance, cov, U_history, X_history, Frequencies_history,  error_history

    def explore(self,true_instance,est_instance,epoch_len,epoch_idx,hyperparams_control=None,past_cov=None):
        dimx, dimu, dimz = est_instance.get_dim()
        input_power = 0
        x = est_instance.get_states_init()
        dt = est_instance.get_time_step()
        n_data = int(self.H/dt)
        u_history, x_history = [],[]
        
        if past_cov is None:
            cov = torch.zeros((dimz,dimz))
        else:
            cov = past_cov
            
        hyperparams_history = []
        for i, p in enumerate(range(epoch_len)):

            if hyperparams_control is None:
                hyperparams = self.optimal_params.random_parameters(n_explore=1)
            else:
                hyperparams = np.array([hyperparams_control[i]])
            
            t_explore = torch.arange(0,self.H,dt)
            t_span = (t_explore[0], t_explore[-1])
            u_hist = self.forcing.control_data(hyperparams[0],t_explore,dimu)
            u_func = self.forcing.control_function(hyperparams[0])
            u_func_partial = partial(u_func)

            x_hist = torch.tensor(solve_ivp(true_instance.true_dynamics, t_span, est_instance.get_states_init(),
                                   t_eval=t_explore,**est_instance.get_int_key(),args=(u_func_partial,)).y.T)
            
            hyperparams_history.append(hyperparams[0])
            u_history.append(u_hist)
            x_history.append(x_hist)
        
        return x_history, u_history, hyperparams_history, input_power, 0, cov
    

class TraceExploration:
    
    def __init__(self,N,H,n_processes,signal_type,amplitude,evaluate,kernel,sim_frac=0.1,ExpObj="A_Optimal",metrics="normalized"):

        self.N = N
        self.H = H
        self.n_processes = n_processes
        self.signal_type = signal_type
        self.amplitude = amplitude
        self.evaluate = evaluate
        self.kernel = kernel
        self.sim_frac = sim_frac
        self.ExpObj = ExpObj
        self.metrics = metrics

        self.optimal_params = hyperparameters_find(self.H,self.n_processes,self.sim_frac,self.signal_type,self.amplitude,self.ExpObj,self.kernel)
        self.forcing = ControlInput(self.signal_type, self.amplitude)

    def system_id(self,true_instance,est_instance,epochs,closed_form,hyperparams_control,verbose=1):
        self.closed_form = closed_form
        x = est_instance.get_states_init()
        self.U_history, self.X_history, self.Frequencies_history, self.error_history = [], [], [], []
        cov = None
        
        for epoch_idx, i in enumerate(range(len(epochs))):     
            if epoch_idx == 0:
                x_history, u_history, frequencies_history, input_power, _, cov = self.explore(true_instance,est_instance,epochs[epoch_idx],epoch_idx,hyperparams_control=hyperparams_control,past_cov=cov)
            else:
                x_history, u_history, frequencies_history, input_power, _, cov = self.explore(true_instance,est_instance,epochs[epoch_idx],epoch_idx,cov)
            
            self.U_history = self.U_history + u_history
            self.X_history = self.X_history + x_history
            self.Frequencies_history = self.Frequencies_history + frequencies_history
            est_instance.update_estimates(self.X_history,self.U_history,closed_form=closed_form)

            # self.error_history.append(true_instance.compute_est_error(est_instance.get_dynamics(), metric=self.metrics))
            self.error_history.append(self.evaluate.compute_est_error(est_instance))
            model = est_instance.get_dynamics().numpy()
            if verbose == 0 or verbose == 1:
                print("Epoch ",str(epoch_idx)," :  loss :",self.error_history[epoch_idx])
            if verbose == 0:
                print("Model Dynamics :\n",np.round(model,3))  
        
        return est_instance, cov, self.U_history, self.X_history, self.Frequencies_history, self.error_history
    
    def explore(self,true_instance,est_instance,epoch_len,epoch_idx,hyperparams_control=None,past_cov=None):
        # self.true_dynamics = true_instance._true_dynamics
        dimx, dimu, dimz = est_instance.get_dim()
        input_power = 0
        x = est_instance.get_states_init()
        dt = est_instance.get_time_step()
        n_data = int(self.H/dt)
        u_history, x_history = [],[]
        
        if past_cov is None:
            cov = torch.zeros((dimz,dimz))
        else:
            cov = past_cov
        
        hyperparams_history = []
        if epoch_idx == 0:
            uniform = RandomExploration(self.N,self.H,self.signal_type,self.amplitude,self.evaluate,self.kernel,metrics=self.metrics)
            return uniform.explore(true_instance,est_instance,epoch_len,epoch_idx,hyperparams_control=hyperparams_control)
        else:    
            for p, i in enumerate(range(epoch_len)):
                
                hyperparams = self.optimal_params.hyperparameters_find(est_instance, self.N, n_data)

                t_explore = torch.arange(0,self.H,dt)
                t_span = (t_explore[0], t_explore[-1])
                u_hist = self.forcing.control_data(hyperparams[0],t_explore,dimu)
                u_func = self.forcing.control_function(hyperparams[0])
                u_func_partial = partial(u_func)

                x_hist = torch.tensor(solve_ivp(true_instance.true_dynamics, t_span, est_instance.get_states_init(),
                                       t_eval=t_explore,**est_instance.get_int_key(),args=(u_func_partial,)).y.T)

                u_history.append(u_hist)
                x_history.append(x_hist)
                hyperparams_history.append(hyperparams[0])

            return x_history, u_history, hyperparams_history, input_power, 0, cov