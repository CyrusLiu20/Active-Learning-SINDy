import torch
import pysindy as ps
import numpy as np
from scipy.integrate import solve_ivp

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

    def dynamics_deri(self,t,x,u_func):
        u = u_func(t)
        phi = self.get_phi(x,u)
        x_dot = torch.matmul(self.A.type(torch.FloatTensor),phi.type(torch.FloatTensor))
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
        phi = self.get_phi(x,u)
        if not(parallel):
            cov = torch.outer(phi.flatten(),phi.flatten())
        else:
            phi = phi.T
            cov = torch.einsum('ik, jk -> ijk',phi,phi)

        return cov 

    def get_phi(self,x,u):
        phi = self.kernel.phi(x,u)
        if len(x.shape) == 1:
            phi = torch.tensor(phi).reshape(-1,1)
        return phi

    def compute_est_error(self,A_est,metrics="default"):
        if metrics == "default":
            error = ((self.A.flatten() - A_est.flatten()) @ (self.A.flatten() - A_est.flatten()))  
        elif metrics == "normalized":
            A_normalized = torch.div((self.A.flatten() - A_est.flatten()),self.A_masked.flatten())
            error = A_normalized @ A_normalized
        else:
            raise ValueError("Unrecognized Model Loss Metrics: {}".format(metrics))
        
        return error
            
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