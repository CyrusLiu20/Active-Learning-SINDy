from time import time
import numpy as np
import casadi as ca

# import warnings
# warnings.filterwarnings("error")

class MPC_Controller:
    
    def __init__(self,controller_parameters):
        
        self.states = controller_parameters['states']
        self.controls = controller_parameters['controls']
        self.states_mapped = controller_parameters['states_mapped']
        self.N = controller_parameters['N']
        self.Q = controller_parameters['Q']
        self.R = controller_parameters['R']
        self.x_lb = controller_parameters['x_lb']
        self.x_ub = controller_parameters['x_ub']
        self.u_lb = controller_parameters['u_lb']
        self.u_ub = controller_parameters['u_ub']
        self.states_init = controller_parameters['states_init']
        self.dt = controller_parameters['dt']
        self.dt_simulate = controller_parameters['dt_simulate']
        self.sim_time = controller_parameters['sim_time']
    
    def initialize(self, A_estimate, A_true):

        self.A_estimate = A_estimate
        self.A_true = A_true
        
        self.X, self.U, self.P, self.n_states, self.n_controls = self.MPC_variables(self.states,self.controls)
        self.f_estimate, self.f_true = self.MPC_model(self.A_estimate,self.A_true,self.states_mapped,self.states,self.controls)
        self.cost_fn, self.g = self.solver_setting(self.X,self.U,self.P,self.f_estimate,self.n_states,self.dt,self.dt_simulate)
        self.solver = self.get_solver(self.cost_fn,self.g,self.X,self.U,self.P,self.sim_time,self.dt)
        self.args = self.solver_contraints(self.x_lb,self.x_ub,self.u_lb,self.u_ub,self.n_states,self.n_controls)
        self.t0, self.t_hist, self.cost_hist, self.u0, self.X0, self.mpc_iter, self.states_hist, self.controls_hist, self.cost_hist, self.times = self.MPC_initialize_parameters(self.n_controls,self.states_init,self.N)
        
    def MPC_run(self,states_target):
        main_loop = time()  # return time in sec
        tolerance = 1e-1
        self.states_target = states_target
        
        try:
            # while (ca.norm_2(states_init - states_target) > tolerance) and (mpc_iter * dt < sim_time):
            while (self.mpc_iter * self.dt < self.sim_time): # Check if stabilises
                t1 = time()

                # MPC Core
                self.args['p'], self.args['x0'] = self.solver_arguments(self.states_init,self.states_target,self.X0,self.u0)
                sol = self.solver(x0=self.args['x0'],lbx=self.args['lbx'],ubx=self.args['ubx'],
                            lbg=self.args['lbg'],ubg=self.args['ubg'],p=self.args['p']) # RK4 Simulation and Optimisation

                self.u_cat = ca.reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N)
                self.X0 = ca.reshape(sol['x'][: self.n_states * (self.N+1)], self.n_states, self.N+1)
                self.cost = sol["f"]

                self.states_hist, self.controls_hist, self.t_hist = self.store_history(self.states_hist,self.controls_hist,self.cost,self.X0,self.u_cat,self.t_hist,self.t0)
                self.X0 = ca.horzcat(self.X0[:, 1:],ca.reshape(self.X0[:, -1], -1, 1))
                self.t0, self.states_init, self.u0 = self.next_timestep(self.dt, self.t0, self.states_init, self.u_cat, self.f_true)


                t2 = time()
                self.times = np.vstack((self.times,t2-t1))
                self.mpc_iter = self.mpc_iter + 1
                if np.any(np.isnan(np.array(self.states_init))):
                    raise ValueError("The States Variable contains NaN values.")

            main_loop_time = time()
            ss_error = ca.norm_2(self.states_init - states_target)
            status = 0
        except ValueError as e:
            print(f"CASADI Solver failed with exception: {e}")
            status = -1

        # print('Total CPU time: %f (s)' % (main_loop_time - main_loop))
        # print('Average CPU iteration time: %f (ms)' % (np.array(self.times).mean() * 1000))
        # # print("Simulation time required for convergence: %f (s)" % t_hist[-1,0])
        # print("Iteration (%d)" % self.mpc_iter)
        # print("final state discrepancy : %f " % ss_error)
        # print("Final State :",np.array(self.states_init).flatten())
        
        return self.t_hist, self.states_hist, self.controls_hist, self.cost_hist, status
        
        

    def MPC_variables(self,states,controls):
        n_states = states.numel()
        n_controls = controls.numel()

        # matrix containing all states over all time steps +1 (each column is a state vector)
        X = ca.SX.sym('X', n_states, self.N + 1)
        U = ca.SX.sym('U', n_controls, self.N)

        # coloumn vector for storing initial state and target state
        P = ca.SX.sym('P', n_states + n_states)

        return X, U, P, n_states, n_controls

    def MPC_model(self,A_estimate,A_true,states_mapped,states,controls):
        RHS_estimate = A_estimate @ states_mapped
        RHS_true = A_true @ states_mapped

        # Chaotic Lorenz system mapping from [x,y,z,u].T to [x,y,z]_dot.T
        f_estimate = ca.Function('f_estimate', [states, controls], [RHS_estimate])
        f_true = ca.Function('f_true', [states, controls], [RHS_true])     

        return f_estimate, f_true

    def solver_setting(self,X,U,P,f_estimate,n_states,dt,dt_simulate):
        cost_fn = 0  # cost function
        g = X[:, 0] - P[:n_states]  # constraints in the equation

        skip_ratio = int(dt/dt_simulate)
        # Runge Kutta 4 numerical scheme
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]

            # Cost function
            cost_fn = cost_fn \
                + (st - P[n_states:]).T @ self.Q @ (st - P[n_states:]) \
                + con.T @ self.R @ con

            st_next = X[:, k+1]
            for i in range(skip_ratio):
                k1 = f_estimate(st, con)
                k2 = f_estimate(st + dt_simulate/2*k1, con)
                k3 = f_estimate(st + dt_simulate/2*k2, con)
                k4 = f_estimate(st + dt_simulate * k3, con)
                st_next_RK4 = st + (dt_simulate / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                st = st_next_RK4

            g = ca.vertcat(g, st_next - st_next_RK4) # equality constraints

        return cost_fn, g

    def get_solver(self,cost_fn,g,X,U,P,sim_time,dt):
        OPT_variables = ca.vertcat(
            X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            U.reshape((-1, 1))
        )
        nlp_prob = {
            'f': cost_fn,
            'x': OPT_variables,
            'g': g,
            'p': P
        }

        opts = {
            'ipopt': {
                'max_iter': int(sim_time/dt),
                'print_level': 0,
                'sb' : 'yes',
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        return solver

    def solver_contraints(self,x_lb,x_ub,u_lb,u_ub,n_states,n_controls):
        lbx = ca.DM.zeros((n_states*(self.N+1) + n_controls*self.N, 1))
        ubx = ca.DM.zeros((n_states*(self.N+1) + n_controls*self.N, 1))

        for i in range(len(x_lb)):
            lbx[i: n_states*(self.N+1): n_states] = x_lb[i]     # state lower bound
            ubx[i: n_states*(self.N+1): n_states] = x_ub[i]      # state upper bound

        for i in range(len(u_lb)):
            lbx[n_states*(self.N+1)+i::n_controls] = u_lb[i]              # controls lower bound for all u
            ubx[n_states*(self.N+1)+i::n_controls] = u_ub[i]                # controls upper bound for all u

        args = {
            'lbg': ca.DM.zeros((n_states*(self.N+1), 1)),  # equality constraints lower bound (= 0)
            'ubg': ca.DM.zeros((n_states*(self.N+1), 1)),  # equality constraints upper bound (= 0)
            'lbx': lbx,
            'ubx': ubx
        }

        return args

    def MPC_initialize_parameters(self,n_controls,states_init,N):
        t0 = 0
        t_hist = ca.DM(t0)
        cost_hist = []

        u0 = ca.DM.zeros((n_controls, N))  # initial control
        X0 = ca.repmat(states_init, 1, N+1) # initial state full

        mpc_iter = 0
        states_hist = np.array(X0.full())
        controls_hist = np.array(u0[:, 0].full())
        times = np.array([[0]]) # Successive computational time

        return t0, t_hist, cost_hist, u0, X0, mpc_iter, states_hist, controls_hist, cost_hist, times

    ######################### MPC Hidden Code #########################
    def next_timestep(self,dt, t0, state_init, u, f):
        f_value = f(state_init, u[:, 0])
        next_state = ca.DM.full(state_init + (dt * f_value))

        t0 = t0 + dt
        u0 = ca.horzcat(
            u[:, 1:],
            ca.reshape(u[:, -1], -1, 1)
        )
        return t0, next_state, u0

    def solver_arguments(self,states_init,states_target,X0,u0):
        args_p = ca.vertcat(
            states_init,    # current state
            states_target   # target state
        )
        # optimisation variable current state
        args_x0 = ca.vertcat(
            ca.reshape(X0, self.n_states*(self.N+1), 1),
            ca.reshape(u0, self.n_controls*self.N, 1)
        )
        return args_p, args_x0

    def store_history(self,states_hist,controls_hist,cost,X0,u,t_hist,t0):
        states_hist = np.dstack((states_hist,np.array(X0.full())))
        controls_hist = np.vstack((controls_hist,np.array(u[:, 0].full())))
        t_hist = np.vstack((t_hist,t0))
        self.cost_hist.append(cost)
        return states_hist, controls_hist, t_hist