import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial

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

class RandomExploration:
    
    def __init__(self,N,H,power_max,objective="random"):

        self.N = N
        self.H = H
        self.objective = objective
        self.power_max = power_max
    
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
            
            error_history.append(true_instance.compute_est_error(est_instance.get_dynamics()))
            model = est_instance.get_dynamics().numpy()
            if verbose == 0 or verbose == 1:
                print("Epoch ",str(epoch_idx)," :  loss :",error_history[epoch_idx])
            if verbose == 0:
                print("Model Dynamics :\n",np.round(model,3))   
        
        return est_instance, cov, U_history, X_history, Frequencies_history,  error_history

    def explore(self,true_instance,est_instance,epoch_len,epoch_idx,frequencies_control=None,past_cov=None):
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
            
        # True dynamics (accelerated)
        def dynamics(t,x,u_func):
            u = u_func(t)
            x_dot = np.zeros((5, 1))
            x_dot[0] = lambda1_sys - d_sys*x[0] - beta_sys*(1 - eta_sys*u)*x[0]*x[1]
            x_dot[1] = beta_sys*(1 - eta_sys*u)*x[1]*x[0] - a_sys*x[1] - p1_sys*x[3]*x[1] - p2_sys*x[4]*x[1]
            x_dot[2] = c2_sys*x[0]*x[1]*x[2] - c2_sys*q_sys*x[1]*x[2] - b2_sys*x[2]
            x_dot[3] = c1_sys*x[1]*x[3] - b1_sys*x[3]
            x_dot[4] = c2_sys*q_sys*x[1]*x[2] - h_sys*x[4]
            return x_dot
            
        frequencies_history = []
        for i, p in enumerate(range(epoch_len)):

            if frequencies_control is None:
                frequency = np.random.uniform(0.5, 15)
            else:
                frequency = frequencies_control[i]
                
            def u_func(t):
                return np.column_stack([np.sin(frequency*t)])
            u_func_partial = partial(u_func)
            
            t_explore = torch.arange(0,self.H,dt)
            t_span = (t_explore[0], t_explore[-1])
            u_hist = torch.sin(frequency*t_explore).reshape(n_data,dimu)
            
            x_hist = torch.tensor(solve_ivp(dynamics, t_span, est_instance.get_states_init(),
                                   t_eval=t_explore,**est_instance.get_int_key(),args=(u_func_partial,)).y.T)

            frequencies_history.append(frequency)
            u_history.append(u_hist)
            x_history.append(x_hist) # Closed form estimate
        
        return x_history, u_history, frequencies_history, input_power, 0, cov
    

class TraceExploration:
    
    def __init__(self,N,H,power_max,sim_frac=0.1,ExpObj="A_Optimal"):

        self.N = N
        self.H = H
        self.power_max = power_max
        self.sim_frac = sim_frac
        self.ExpObj = ExpObj
      
    def system_id(self,true_instance,est_instance,epochs,closed_form,frequencies_control,verbose=1):
        self.closed_form = closed_form
        x = est_instance.get_states_init()
        self.U_history, self.X_history, self.Frequencies_history, self.error_history = [], [], [], []
        cov = None
        
        for epoch_idx, i in enumerate(range(len(epochs))):     
            if epoch_idx == 0:
                x_history, u_history, frequencies_history, input_power, _, cov = self.explore(true_instance,est_instance,epochs[epoch_idx],epoch_idx,frequencies_control=frequencies_control,past_cov=cov)
            else:
                x_history, u_history, frequencies_history, input_power, _, cov = self.explore(true_instance,est_instance,epochs[epoch_idx],epoch_idx,cov)
            
            self.U_history = self.U_history + u_history
            self.X_history = self.X_history + x_history
            self.Frequencies_history = self.Frequencies_history + frequencies_history
            est_instance.update_estimates(self.X_history,self.U_history,closed_form=closed_form)
            
            self.error_history.append(true_instance.compute_est_error(est_instance.get_dynamics()))
            model = est_instance.get_dynamics().numpy()
            if verbose == 0 or verbose == 1:
                print("Epoch ",str(epoch_idx)," :  loss :",self.error_history[epoch_idx])
            if verbose == 0:
                print("Model Dynamics :\n",np.round(model,3))  
        
        return est_instance, cov, self.U_history, self.X_history, self.Frequencies_history, self.error_history
    
    def explore(self,true_instance,est_instance,epoch_len,epoch_idx,frequencies_control=None,past_cov=None):
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
            
        # True dynamics (accelerated)
        def dynamics(t,x,u_func):
            u = u_func(t)
            x_dot = np.zeros((5, 1))
            x_dot[0] = lambda1_sys - d_sys*x[0] - beta_sys*(1 - eta_sys*u)*x[0]*x[1]
            x_dot[1] = beta_sys*(1 - eta_sys*u)*x[1]*x[0] - a_sys*x[1] - p1_sys*x[3]*x[1] - p2_sys*x[4]*x[1]
            x_dot[2] = c2_sys*x[0]*x[1]*x[2] - c2_sys*q_sys*x[1]*x[2] - b2_sys*x[2]
            x_dot[3] = c1_sys*x[1]*x[3] - b1_sys*x[3]
            x_dot[4] = c2_sys*q_sys*x[1]*x[2] - h_sys*x[4]
            return x_dot
        
        frequencies_history = []
        if epoch_idx == 0:
            uniform = RandomExploration(self.N,self.H,self.power_max,objective="random")
            return uniform.explore(true_instance,est_instance,epoch_len,epoch_idx,frequencies_control=frequencies_control)
        else:    
            for p, i in enumerate(range(epoch_len)):
                
                frequency = self.frequency_find(est_instance,n_data)
                
                def u_func(t):
                    return np.column_stack([np.sin(frequency*t)])
                u_func_partial = partial(u_func)

                t_explore = torch.arange(0,self.H,dt)
                t_span = (t_explore[0], t_explore[-1])
                u_hist = torch.sin(frequency*t_explore).reshape(n_data,dimu)

                x_hist = torch.tensor(solve_ivp(dynamics, t_span, est_instance.get_states_init(),
                                       t_eval=t_explore,**est_instance.get_int_key(),args=(u_func_partial,)).y.T)

                u_history.append(u_hist)
                x_history.append(x_hist)
                frequencies_history.append(frequency)

            return x_history, u_history, frequencies_history, input_power, 0, cov
    
    def frequency_find(self,est_instance,n_data):
        
        frequencies = np.random.uniform(0.5, 15,size=self.N)
        costs = torch.zeros_like(torch.tensor(frequencies))
        
        for i, frequency in enumerate(frequencies):
            def u_func(t):
                return np.column_stack([np.sin(frequency*t)])
            u_func_partial = partial(u_func)

            
            t_explore = torch.arange(0,self.H*self.sim_frac,est_instance.dt)
            t_span = (t_explore[0], t_explore[-1])
            u_hist = torch.sin(frequency*t_explore).reshape(len(t_explore),est_instance.dimu)

            x_hist = torch.tensor(solve_ivp(est_instance.dynamics_deri, t_span, est_instance.get_states_init(),
                                   t_eval=t_explore,**est_instance.get_int_key(),args=(u_func_partial,)).y.T)
            if self.ExpObj == "A_Optimal":           
                costs[i] = self.A_optimal_cost(x_hist,u_hist,est_instance,ignore_frac=0)
            elif self.ExpObj == "E_Optimal":
                costs[i] =  self.E_optimal_cost(x_hist,u_hist,est_instance,ignore_frac=0) 
            elif self.ExpObj == "D_Optimal":
                costs[i] =  self.D_optimal_cost(x_hist,u_hist,est_instance,ignore_frac=0)          
        
        if self.ExpObj == "A_Optimal":
            # FW approximation for minimising tr(inv(cov))
            idx = torch.argmax(costs)
        elif self.ExpObj == "E_Optimal":
            # Maximising the minimum eigenvalue
            idx = torch.argmax(costs)
        elif self.ExpObj == "D_Optimal":
            # Minimising the log determinant of covariance
            idx = torch.argmin(costs)

        print("Frequency Chosen : ", frequencies[idx])

        return frequencies[idx]
    
    def A_optimal_cost(self,x_horizon,u_horizon,est_instance,ignore_frac=0):
        
        n_data = len(u_horizon)
        covs = torch.zeros_like(est_instance.get_cov(x_horizon[0],u_horizon[0]))
        for i in range(int(ignore_frac*n_data),n_data):
            covs = covs + est_instance.get_cov(x_horizon[i],u_horizon[i])
        
        R = self.hess_to_cost(covs,est_instance.dimx)
        cost = torch.trace(R.type(torch.FloatTensor) @ covs.type(torch.FloatTensor))
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
    
    def E_optimal_cost(self,x_horizon,u_horizon,est_instance,ignore_frac=0):
        n_data = len(u_horizon)
        covs = torch.zeros_like(est_instance.get_cov(x_horizon[0],u_horizon[0]))
        for i in range(int(ignore_frac*n_data),n_data):
            covs = covs + est_instance.get_cov(x_horizon[i],u_horizon[i])

        e,_ = torch.linalg.eig(covs)
        cost = torch.min(torch.real(e))
        return cost
    
    def D_optimal_cost(self,x_horizon,u_horizon,est_instance,ignore_frac=0):
        n_data = len(u_horizon)
        covs = torch.zeros_like(est_instance.get_cov(x_horizon[0],u_horizon[0]))
        for i in range(int(ignore_frac*n_data),n_data):
            covs = covs + est_instance.get_cov(x_horizon[i],u_horizon[i])

        cost = torch.log(torch.linalg.det(covs))
        return cost