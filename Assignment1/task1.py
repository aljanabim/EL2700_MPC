from model import Pendulum
from controller import Controller
from simulation import EmbeddedSimEnvironment
import numpy as np
# Create pendulum and controller objects
pendulum = Pendulum()

# Get the system discrete-time dynamics
A, B, C = pendulum.get_discrete_system_matrices_at_eq()

# Get control gains
p1 = np.random.uniform(0.8, 1, size=(100000000)).tolist()
p2 = np.random.uniform(0.8, 1, size=(100000000)).tolist()
p3 = np.random.uniform(0.8, 1, size=(100000000)).tolist()
p4 = np.random.uniform(0.8, 1, size=(100000000)).tolist()

for i in range(len(p1)):
    # print(i)
    ctl = Controller()
    ctl.set_poles(p1[i], p2[i], p3[i], p4[i])
    ctl.set_system(A, B, C)
    K = ctl.get_closed_loop_gain()
    lr = ctl.get_feedforward_gain(K)

    # Initialize simulation environment
    sim_env = EmbeddedSimEnvironment(model=pendulum,
                                     dynamics=pendulum.discrete_time_dynamics,
                                     controller=ctl.control_law,
                                     time=10)
    t, y, u = sim_env.run([0, 0, 0, 0])
    x1 = y[0, :]
    x3 = 180 / np.pi * y[2, :]
    if 9 < x1[-1] and x1[-1] < 10 and np.max(x3) < 10 and np.min(x3) > -10:
        print(f"Doing the {i}:th iteration")
        print(f"p1={p1[i]} | p2={p2[i]} | p3={p3[i]} | p4={p4[i]}")
        print(x1[-1], " [m] ", np.max(x3), " [deg]")


# p1 = 0.9406153029109776 | p2 = 0.9364636826115367 | p3 = 0.9245508021924024 | p4 = 0.9349221655550173
# 9.073511443660959  [m]  12.477053611118087  [deg]

# p1 = 0.9378831014663548 | p2 = 0.9336227701310827 | p3 = 0.9452702170377756 | p4 = 0.9175734886549203
# 9.038568582647915  [m]  12.43134136915953  [deg]

# p1 = 0.9203441907036086 | p2 = 0.9408387623149864 | p3 = 0.9453434874553869 | p4 = 0.9290681250442654
# 9.019026167063686  [m]  12.269755793111818  [deg]

# p1 = 0.9431394297545886 | p2 = 0.9314423396835586 | p3 = 0.9362427935850063 | p4 = 0.9299153844710516
# 9.011198993229666  [m]  11.924880116061082  [deg]


# Enable model disturbance for second simulation environment
pendulum.enable_disturbance(w=0.01)
# sim_env_with_disturbance = EmbeddedSimEnvironment(model=pendulum,
#                                                   dynamics=pendulum.continuous_time_nonlinear_dynamics,
#                                                   controller=ctl.control_law,
#                                                   time=20)

# Also returns time and state evolution
# t, y, u = sim_env_with_disturbance.run([0, 0, 0, 0])
