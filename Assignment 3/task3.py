import numpy as np

from model import Pendulum
from dlqr import DLQR
from simulation import EmbeddedSimEnvironment

# Create pendulum and controller objects
pendulum = Pendulum()

# Get the system discrete-time dynamics
A, B, Bw, C = pendulum.get_discrete_system_matrices_at_eq()
ctl = DLQR(A, B, C)

# Get control gains
K, P = ctl.get_lqr_gain(Q= np.diag([0, 0, 0, 0]), 
                        R= 0)

# Get feeforward gain
lr = ctl.get_feedforward_gain(K)

# Part I - no disturbance
sim_env = EmbeddedSimEnvironment(model=pendulum, 
                                dynamics=pendulum.discrete_time_dynamics,
                                controller=ctl.feedfwd_feedback,
                                time = 20)
sim_env.set_window(10)
t, y, u = sim_env.run([0,0,0,0])

# Part I - with disturbance
pendulum.enable_disturbance(w=0.01)  
sim_env_with_disturbance = EmbeddedSimEnvironment(model=pendulum, 
                                dynamics=pendulum.discrete_time_dynamics,
                                controller=ctl.feedfwd_feedback,
                                time = 20)
sim_env_with_disturbance.set_window(10)
t, y, u = sim_env_with_disturbance.run([0,0,0,0])


### Part II
Ai, Bi, Bwi, Ci = pendulum.get_augmented_discrete_system()
ctl.set_system(Ai, Bi, Ci)
K, P = ctl.get_lqr_gain(Q= np.diag([0, 0, 0, 0]), 
                        R= 0)

# Get feeforward gain              
ctl.set_lr(lr)     

pendulum.enable_disturbance(w=0.01)  
sim_env_with_disturbance = EmbeddedSimEnvironment(model=pendulum, 
                                dynamics=pendulum.pendulum_augmented_dynamics,
                                controller=ctl.lqr_ff_fb_integrator,
                                time = 20)
sim_env_with_disturbance.set_window(10)
t, y, u = sim_env_with_disturbance.run([0,0,0,0,0])

### Part III
# Output feedback
C = np.array([[1,0,0,0]])
C = np.array([[1,0,0,0],
              [0,0,1,0]])

Qp = np.eye(4)
Rn = np.eye(np.size(C,0))
pendulum.set_kf_params(C,Qn,Rn)
pendulum.init_kf()

sim_env_with_disturbance_estimated = EmbeddedSimEnvironment(model=pendulum, 
                                dynamics=pendulum.pendulum_augmented_dynamics,
                                controller=ctl.lqr_ff_fb_integrator,
                                time = 20)
sim_env_with_disturbance_estimated.set_estimator(True)
sim_env_with_disturbance_estimated.set_window(10)
t, y, u = sim_env_with_disturbance_estimated.run([0,0,0,0,0])