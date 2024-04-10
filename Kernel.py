import torch
import numpy as np
from itertools import combinations_with_replacement

# HIV System Non-linearity
class HIVKernel:
    
    def __init__(self):
        pass
        
    def phi(self,x,u):
        if not(torch.is_tensor(x)):
            x = torch.tensor(x)
        if not(torch.is_tensor(u)):
            u = torch.tensor(u)
        # if x.numel() < 5+1:
        ab = x[0]*x[1]
        phi_out = torch.vstack([torch.tensor([1]),x.reshape(-1,1),u.reshape(-1,1)])
        phi_out = torch.vstack([phi_out,torch.tensor([ab,x[1]*x[2],x[1]*x[3],x[1]*x[4],\
                                                        ab*x[2],ab*x[3],ab*x[4],\
                                                        ab*u[0]]).reshape(-1,1)])

        return phi_out

    def phi_np(self, x, u):
        if not isinstance(x, np.ndarray) or not isinstance(u, np.ndarray):
            x = np.array(x)
            u = np.array(u)
            
        ab = x[0] * x[1]
        phi_out = np.vstack([np.array([1]), x.reshape(-1, 1), u.reshape(-1, 1)])
        phi_out = np.vstack([phi_out, np.array([ab, x[1] * x[2], x[1] * x[3], x[1] * x[4],
                                                ab * x[2], ab * x[3], ab * x[4],
                                                ab * u[0][0]]).reshape(-1, 1)])
        
        return phi_out
    
# HIV System Non-linearity
class F8Kernel:
    
    def __init__(self, degree, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias

    def phi(self,x,u):
        if not(torch.is_tensor(x)):
            x = torch.tensor(x)
        if not(torch.is_tensor(u)):
            u = torch.tensor(u)

        phi_out = torch.tensor([1, x[0], x[1], x[2], u[0], x[0]**2, x[0]*x[1], x[0]*x[2], x[0]*u[0], x[1]**2, x[1]*x[2], x[1]*u[0], x[2]**2, x[2]*u[0], u[0]**2,
                                x[0]**3, x[0]**2*x[1], x[0]**2*x[2], x[0]**2*u[0], x[0]*x[1]**2, x[0]*x[1]*x[2], x[0]*x[1]*u[0], x[0]*x[2]**2, x[0]*x[2]*u[0], x[0]*u[0]**2, 
                                x[1]**3, x[1]**2*x[2], x[1]**2*u[0],x[1]*x[2]**2, x[1]*x[2]*u[0], x[1]*u[0]**2, x[2]**3, x[2]**2*u[0], x[2]*u[0]**2, u[0]**3]).reshape(-1, 1)
        return phi_out        

    def phi_np(self, x, u):
        if not isinstance(x, np.ndarray) or not isinstance(u, np.ndarray):
            x = np.array(x)
            u = np.array(u)
        
        # Third Degree
        phi_out = np.array([1, x[0], x[1], x[2], u[0][0], x[0]**2, x[0]*x[1], x[0]*x[2], x[0]*u[0][0], x[1]**2, x[1]*x[2], x[1]*u[0][0], x[2]**2, x[2]*u[0][0], u[0][0]**2,
                            x[0]**3, x[0]**2*x[1], x[0]**2*x[2], x[0]**2*u[0][0], x[0]*x[1]**2, x[0]*x[1]*x[2], x[0]*x[1]*u[0][0], x[0]*x[2]**2, x[0]*x[2]*u[0][0], x[0]*u[0][0]**2, 
                            x[1]**3, x[1]**2*x[2], x[1]**2*u[0][0],x[1]*x[2]**2, x[1]*x[2]*u[0][0], x[1]*u[0][0]**2, x[2]**3, x[2]**2*u[0][0], x[2]*u[0][0]**2, u[0][0]**3]).reshape(-1, 1)
        return phi_out


class SymbolicPolynomialLibrary:
    def __init__(self):
        pass

    def fit(self, variables, max_degree, include_bias=True):
        self.variables = variables
        self.max_degree = max_degree
        self.include_bias = include_bias
        self.feature_expressions = []

        if self.include_bias:
            self.feature_expressions.append(1)  # Include a bias term (constant)

        for deg in range(1, self.max_degree + 1):
            for combo in combinations_with_replacement(self.variables, deg):
                feature_expression = 1
                for var in combo:
                    feature_expression *= var
                self.feature_expressions.append(feature_expression)

        return self.feature_expressions
    
    def get_feature_map(self):
        return self.feature_expressions
    
    def print_feature_names(self):
        print("Symbolic terms of the feature library (degree %d)" % self.max_degree)
        for feature in self.feature_expressions:
            print(feature, end=",")
        print()