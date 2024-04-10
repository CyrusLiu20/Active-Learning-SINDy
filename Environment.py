import torch
import pysindy as ps
import numpy as np
import casadi as ca
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from functools import partial

from controller import MPC_Controller

class Environment:
    
    def __init__(self,A,dimu,noise_cov,states_init,kernel,dt,H=10):
        self.A = A
        self.kernel = kernel
        self.states_init = states_init
        self.dimx = A.shape[0]
        self.dimu = dimu
        self.dim_phi = A.shape[1]
        self.dt = dt
        self.H = H
        self.discretize = 10
        self.noise_cov = noise_cov

        self.A_masked = torch.where(A == 0, 1, A)

        self.data_cov = torch.zeros(self.dim_phi,self.dim_phi)
        self.data_process = torch.zeros(self.dim_phi,self.dimx)
        
        if torch.is_tensor(noise_cov) is False:
            noise_cov = torch.tensor(noise_cov, dtype=torch.float32)
        if len(noise_cov.shape) == 0:
            self.noise_std = torch.sqrt(noise_cov) * torch.eye(self.dimx)
        else:
            U,S,V = torch.svd(noise_cov)
            self.noise_std = U @ torch.diag(torch.sqrt(S)) @ V.T
        
        self.integrator_keywords = {}
        self.integrator_keywords["rtol"] = 1e-12
        self.integrator_keywords["method"] = "LSODA"
        self.integrator_keywords["atol"] = 1e-12
    
    
    def dynamics(self,x,u,noiseless=True):
        phi = self.get_phi(x,u)

        if noiseless:
            # x_dot = torch.tensor(self.A).type(torch.DoubleTensor) @ phi.type(torch.DoubleTensor)
            if torch.tensor(x).numel() < self.dimx+1:
                x_dot = torch.matmul(self.A,phi)
            else:
                x_dot = torch.matmul(self.A,phi.T)
                
        else:
            if torch.tensor(x).numel() < self.dimx+1:
                noise = self.noise_std @ torch.randn(self.dimx)
                x_dot = torch.matmul(self.A,phi) + noise.reshape(-1,1)
            else:
                noise = self.noise_std @ torch.randn(self.dimx,x.shape[1])
                x_dot = torch.matmul(self.A.T,phi.T) + noise

        # if not(derivative):
        return x.reshape_as(x_dot) + self.dt*x_dot
        # if derivative:
        #     return x_dot
    
    def dynamics_con(self,x,u,noiseless=True):
        
        def dynamics(t,x,u):
            phi = self.get_phi(x,u)
            x_dot = torch.matmul(self.A.type(torch.FloatTensor),phi.type(torch.FloatTensor))
            return x_dot
        
        t_debug = torch.linspace(0,self.dt,self.discretize)
        t_span = (t_debug[0], t_debug[-1])
        x = solve_ivp(dynamics, t_span, x,t_eval=t_debug,**self.integrator_keywords,args=(u,)).y.T
        return torch.tensor(x[-1]).reshape(self.dimx,-1)

    # def dynamics_deri(self,t,x,u_func):
    #     u = u_func(t)
    #     phi = self.get_phi(x,u)
    #     x_dot = torch.matmul(self.A.type(torch.FloatTensor),phi.type(torch.FloatTensor))
    #     # x_dot = np.array(self.A) @ phi
    #     return x_dot

    def dynamics_deri(self,t,x,u_func):
        u = u_func(t)
        phi = self.get_phi_np(x,u)
        x_dot = np.matmul(np.array(self.A),phi)
        return x_dot
    
    def update_estimates(self,states,controls,closed_form=True):
        
        if closed_form:
            self.update_parameter_estimates(states,controls)
        else:
            model = ps.SINDy(feature_names=self.feature_names,optimizer=self.optimizer,feature_library=self.library)
            model.fit(x=states,u=controls,t=self.dt,multiple_trajectories=True)
            self.A = torch.tensor(model.coefficients()).type(torch.FloatTensor)

    def update_parameter_estimates(self,states,inputs):
        T = len(inputs)
        for t in range(T):
            for h in range(self.H):
                z = self.get_phi(states[t][h,:],inputs[t][h,:]).flatten()
                self.data_cov += torch.outer(z,z)
                self.data_process += torch.outer(z,states[t][h+1,:])
        
        self.A = torch.linalg.inv(self.data_cov + 0.001*torch.eye(self.dim_phi)) @ self.data_process
        
    def get_cov(self,x,u,parallel=False):
        # phi = self.get_phi_np(x,u)
        phi = self.get_phi(x,u)
        if not(parallel):
            cov = torch.outer(phi.flatten(),phi.flatten())
            # cov = np.outer(phi.flatten(),phi.flatten())
        else:
            phi = phi.T
            cov = torch.einsum('ik, jk -> ijk',phi,phi)

        return cov 
    
    def get_hat(self,x,u,parallel=False):
        # phi = self.get_phi_np(x,u)
        phi = self.get_phi(x,u)
        if not(parallel):
            hat = phi.T@torch.outer(phi.flatten(),phi.flatten())@phi
        else:
            pass

        print(hat)
        return hat

    def get_phi(self,x,u):
        phi = self.kernel.phi(x,u)
        if len(x.shape) == 1:
            phi = phi.clone().detach().reshape(-1,1)
        return phi

    def get_phi_np(self,x,u):
        phi = self.kernel.phi_np(x,u)
        if len(x.shape) == 1:
            phi = np.array(phi).reshape(-1, 1)
        return phi
            
    def reset_dynamics(self):
        self.A = torch.zeros_like(self.A)
    
    def set_dynamics(self,A):
        self.A = A
        
    def set_dynamics_flat(self,A):
        self.A = A.reshape(self.dimx,self.dim_phi)
        
    def get_dynamics(self):
        return self.A

    def get_states_init(self):
        return self.states_init
    
    def set_states_init(self,states_init):
        self.states_init = states_init
    
    def get_dim(self):
        return self.dimx, self.dimu, self.dim_phi

    def get_time_step(self):
        return self.dt
    
    def get_int_key(self):
        return self.integrator_keywords
    
    def get_all_params(self):
        return (self.A,self.dimu,self.noise_cov,self.states_init,self.dt)

class HIVEnvironment(Environment):

    def __init__(self,A,dimu,noise_cov,states_init,kernel,dt,H=10):
        super().__init__(A,dimu,noise_cov,states_init,kernel,dt,H)
        self.optimizer = ps.STLSQ(threshold=0.005)
    
        self.feature_names = ["a","b","c","d","e","u"]
        
        library_linear = ps.PolynomialLibrary(degree=1,include_bias=True) # Feature Library
        non_linear_functions1 = [lambda x,y: x*y]
        library_ab = ps.CustomLibrary(library_functions=non_linear_functions1)
        library_bc = ps.CustomLibrary(library_functions=non_linear_functions1) 
        library_bd = ps.CustomLibrary(library_functions=non_linear_functions1)
        library_be = ps.CustomLibrary(library_functions=non_linear_functions1) 
        
        non_linear_functions2 = [lambda x,y,z: x*y*z]
        library_abc = ps.CustomLibrary(library_functions=non_linear_functions2)         
        library_abd = ps.CustomLibrary(library_functions=non_linear_functions2)         
        library_abe = ps.CustomLibrary(library_functions=non_linear_functions2)         
        library_abu = ps.CustomLibrary(library_functions=non_linear_functions2)         
        
        
        n_libraries = 9
        inputs_temp = np.tile([0, 1, 2, 3, 4, 5], n_libraries)
        inputs_per_library = np.reshape(inputs_temp, (n_libraries, 6))
        inputs_per_library[1, 2], inputs_per_library[1, 3], inputs_per_library[1, 4], inputs_per_library[1, 5] = 1,1,1,1
        inputs_per_library[2, 0], inputs_per_library[2, 3], inputs_per_library[2, 4], inputs_per_library[2, 5] = 1,1,1,1
        inputs_per_library[3, 0], inputs_per_library[3, 2], inputs_per_library[3, 4], inputs_per_library[3, 5] = 1,1,1,1
        inputs_per_library[4, 0], inputs_per_library[4, 2], inputs_per_library[4, 3], inputs_per_library[4, 5] = 1,1,1,1
        
        inputs_per_library[5, 3], inputs_per_library[5, 4], inputs_per_library[5, 5] = 1,1,1
        inputs_per_library[6, 2], inputs_per_library[6, 4], inputs_per_library[6, 5] = 1,1,1
        inputs_per_library[7, 2], inputs_per_library[7, 3], inputs_per_library[7, 5] = 1,1,1
        inputs_per_library[8, 2], inputs_per_library[8, 3], inputs_per_library[8, 4] = 1,1,1
        library = [library_linear,library_ab,library_bc,library_bd,library_be,
                   library_abc,library_abd,library_abe,library_abu]
        
        self.library = ps.GeneralizedLibrary(library,inputs_per_library=inputs_per_library)


        self.lambda1 = 1
        self.d = 0.1
        self.beta = 1
        self.a = 0.2
        self.p1 = 1
        self.p2 = 1
        self.c1 = 0.03
        self.c2 = 0.06
        self.b1 = 0.1
        self.b2 = 0.01
        self.q = 0.5
        self.h = 0.1 
        self.eta = 0.9799 

    def true_dynamics(self,t,x,u_func):
        u = u_func(t)
        x_dot = np.zeros((5, 1))
        x_dot[0] = self.lambda1 - self.d*x[0] - self.beta*(1 - self.eta*u)*x[0]*x[1]
        x_dot[1] = self.beta*(1 - self.eta*u)*x[1]*x[0] - self.a*x[1] - self.p1*x[3]*x[1] - self.p2*x[4]*x[1]
        x_dot[2] = self.c2*x[0]*x[1]*x[2] - self.c2*self.q*x[1]*x[2] - self.b2*x[2]
        x_dot[3] = self.c1*x[1]*x[3] - self.b1*x[3]
        x_dot[4] = self.c2*self.q*x[1]*x[2] - self.h*x[4]
        return x_dot


class F8Environment(Environment):

    def __init__(self,A,dimu,noise_cov,states_init,kernel,dt,H=10):
        super().__init__(A,dimu,noise_cov,states_init,kernel,dt,H)
        self.optimizer = ps.STLSQ(threshold=0.01) 
        self.feature_names = ["x0","x1","x2","u"]
        library_linear = ps.PolynomialLibrary(degree=3,include_bias=True) # Feature Library      
        self.library = ps.GeneralizedLibrary([library_linear])

        self.integrator_keywords = {}
        self.integrator_keywords["rtol"] = 1e-10
        self.integrator_keywords["method"] = "LSODA"
        self.integrator_keywords["atol"] = 1e-10

    def true_dynamics(self,t,x,u_func):
        u = u_func(t)
        x_dot = np.zeros((3, 1))
        x_dot[0] = -0.877*x[0] + x[2] - 0.088*x[0]*x[2] + 0.47*x[0]**2 - 0.019*x[1]**2 - x[0]**2*x[2] + 3.846*x[0]**3 - 0.215*u + 0.28*x[0]**2*u + 0.63*u**3 + 0.47*x[0]*u**2
        x_dot[1] = x[2]
        x_dot[2] = -4.208*x[0] - 0.396*x[2] - 0.47*x[0]**2 - 3.564*x[0]**3 - 20.967*u + 46*x[0]*u**2 + 61.4*u**3 + 6.265*x[0]**2*u
        
        return x_dot

class ModelEvaluate:
 
    def __init__(self, true_instance, control_input, metrics, prediction_parameters, controller_parameters):
        self.A_true = true_instance.get_dynamics()
        self.dimx, self.dimu, self.dim_phi = true_instance.get_dim()
        self.integrator_keywords = true_instance.integrator_keywords

        self.metrics = metrics

        self.A_masked = torch.where(self.A_true == 0, 1, self.A_true)

        self.states_init = prediction_parameters['states_init']
        self.sim_time = prediction_parameters['sim_time']
        self.forcing = control_input
        self.hyperparams = prediction_parameters['hyperparams']

        self.prediction_metrics = ["mape", "mae", "mse", "r2"]
        self.x_true, _ = self.prediction(true_instance, true=True)
        self.controller_metrics = ["mape_con", "mae_con", "mse_con", "r2_con", "lq_cost"]
        self.controller_parameters = controller_parameters
        self.controller = MPC_Controller(controller_parameters=self.controller_parameters)
        self.states_target = ca.DM(self.controller_parameters['states_target'])

    def compute_est_error(self,est_instance):

        A_est = est_instance.get_dynamics()
        error_all = []
        if any(metric in self.metrics for metric in self.prediction_metrics):
            x_est, status = self.prediction(est_instance, true=False)
        if any(metric in self.metrics for metric in self.controller_metrics):
            x_con, u_con, status_con = self.reference_track(est_instance)
            n = len(x_con)
            x_con_true = np.repeat(self.states_target.full().T, n, axis=0)
        for metric in self.metrics:
            if metric == "default":
                error = ((self.A_true.flatten() - A_est.flatten()) @ (self.A_true.flatten() - A_est.flatten()))  
            elif metric == "normalized":
                A_normalized = torch.div((self.A_true.flatten() - A_est.flatten()),self.A_masked.flatten())
                error = A_normalized @ A_normalized

            elif metric == "mape":
                error = mean_absolute_percentage_error(y_true=self.x_true, y_pred=x_est) if status == 0 else float("nan")       
            elif metric == "mae":
                error = mean_absolute_error(y_true=self.x_true, y_pred=x_est) if status == 0 else float("nan")
            elif metric == "mse":
                error = mean_squared_error(y_true=self.x_true, y_pred=x_est) if status == 0 else float("nan")
            elif metric == "r2":
                error = r2_score(y_true=self.x_true, y_pred=x_est) if status == 0 else float("nan")

            elif metric == "mape_con":
                error = mean_absolute_percentage_error(y_true=x_con_true, y_pred=x_con) if status_con == 0 else float("nan")
            elif metric == "lq_cost":
                error = self.controller_cost(x_con, u_con, x_con_true)
            else:
                raise ValueError("Unrecognized Model Loss Metrics: {}".format(self.metric))
        
            error = error if np.abs(error) < 1e10 else float("nan")
            error_all.append(error)

        return error_all
    
    def prediction(self, instance, true=False):
        dt = instance.get_time_step()
        t_eval = torch.arange(0,self.sim_time,dt)
        t_span = (t_eval[0], t_eval[-1])
        u_func = self.forcing.control_function(hyperparams=self.hyperparams)
        u_func_partial = partial(u_func)
        if true:
            sol = solve_ivp(instance.true_dynamics, t_span, self.states_init,
                                    t_eval=t_eval,**self.integrator_keywords,args=(u_func_partial,))
        else:    
            sol = solve_ivp(instance.dynamics_deri, t_span, self.states_init,
                                    t_eval=t_eval,**self.integrator_keywords,args=(u_func_partial,))
        x_hist = torch.tensor(sol.y.T)
        status = sol.status
        return x_hist, status
    
    def reference_track(self, instance):
        self.controller.initialize(np.array(instance.get_dynamics()), np.array(self.A_true))
        t_hist, states_hist, controls_hist, cost_hist, status = self.controller.MPC_run(states_target=self.states_target)
        x_con = self.states_retrieve(states_hist).T
        u_con = self.controls_retrieve(controls_hist)

        return x_con, u_con, status

    def controller_cost(self, x_con, u_con, x_con_true):
        x_error = x_con - x_con_true
        cost = np.trace(x_error @ self.controller_parameters["Q"] @ x_error.T) + np.trace(u_con @ self.controller_parameters["R"] @ u_con.T)
        cost = cost/len(x_con)
        return cost


    def states_retrieve(self, states_hist):
        data = np.zeros((states_hist.shape[0],states_hist.shape[2]))
        for i in range(states_hist.shape[2]):
            for j in range(states_hist.shape[0]):
                data[j,i] = states_hist[j, 0, i]
        return data

    def controls_retrieve(self, controls_hist):
        data = np.zeros((controls_hist.shape[0],controls_hist.shape[1]))
        for i in range(controls_hist.shape[0]):
            data[i] = controls_hist[i, 0]

        return data