import numpy as np
import casadi as ca
from matplotlib import pyplot as plt


def PAS():
    # X will be the number of tickets
    # X = [x1 x2] : x1 - first class tickets; x2 - second class tickets
    lb_x1 = 20
    lb_x2 = 35

    profit_x1 = 2500
    profit_x2 = 2000

    max_passengers = 130
    # x1 + x2 <= 130 -> [1 1] @ [x1 x2] <= 130
    A = ca.DM.ones(1, 2)
    ub_a = max_passengers
    lb_a = lb_x1 + lb_x2

    ub_x = np.inf * np.ones((2))
    lb_x = -np.inf * np.ones((2))
    lb_x[0] = lb_x1
    lb_x[1] = lb_x2

    H = ca.DM.zeros(2, 2)
    g = np.zeros((2, 1))
    g[0, 0] = profit_x1
    g[1, 0] = profit_x2
    g = -1 * g

    print(H.shape)
    qp = {'h': H.sparsity(), "a": A.sparsity()}
    S = ca.conic("S", 'osqp', qp)
    # S = ca.conic("S", 'qpoases', qp)
    r = S(h=H, g=g, a=A, lbx=lb_x, ubx=ub_x, lba=lb_a, uba=ub_a)

    print(f'r[x]= {r["x"]}')


def hanging_chain(N):
    m = 4
    ki = 1000
    gc = 9.81

    # x = [y1 z1 y2 z2 ... yn zn]
    #        1     2         n
    H = ca.DM.zeros(2 * N, 2 * N)
    for i in range(0, 2 * N - 2):
        H[i, i + 2] = -1
        H[i + 2, i] = -1

    for i in range(0, 2 * N):
        H[i, i] = 2.0

    H[0, 0] = H[1, 1] = H[-2, -2] = H[-1, -1] = 1
    H = ki * H

    g = ca.DM.zeros(2 * N)
    g[1:-2:2] = gc * m

    # Initialize the lower bound
    lbx = -np.inf * np.ones(2 * N)
    ubx = np.inf * np.ones(2 * N)

    lbx[0] = ubx[0] = -2
    lbx[1] = ubx[1] = 1
    lbx[-2] = ubx[-2] = 2
    lbx[-1] = ubx[-1] = 1
    # lbx[3:-2:2] = 0.5

    # Tilted ground constraints
    A = ca.DM.zeros(N, 2 * N)
    for k in range(N):
        A[k, 2 * k] = -0.1
        A[k, 2 * k + 1] = 0.1

    lba = 0.5 * ca.DM.ones(N, 1)
    uba = np.inf * ca.DM.ones(N, 1)

    # print("Bound for our sysyem \n LBx: ", lbx, "\nUBx", ubx)
    # qp = {'h': H.sparsity()}
    # S = ca.conic('hc', 'osqp', qp)
    # sol = S(h=H, g=g, lbx=lbx, ubx=ubx)

    print("Bound for our sysyem \n LBx: ", lbx, "\nUBx", ubx)
    qp = {'h': H.sparsity(), 'a': A.sparsity()}
    S = ca.conic('hc', 'osqp', qp)
    sol = S(h=H, g=g, a=A, lbx=lbx, ubx=ubx, lba=lba, uba=uba)

    x_opt = sol['x']
    Y0 = x_opt[0::2]
    Z0 = x_opt[1::2]
    plt.plot(Y0, Z0, 'b-o')
    plt.show()


# PAS()
hanging_chain(10)
