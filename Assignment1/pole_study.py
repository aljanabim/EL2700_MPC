from model import Pendulum
import numpy as np
from scipy.linalg import expm, eig
import casadi as ca
from matplotlib import pyplot as plt
plt.style.use('seaborn-darkgrid')

pendulum = Pendulum()
Ac = pendulum.get_Ac()
Ac_poles = eig(Ac)[0]
h_space = np.linspace(0.02, 1, 1000)

re = []
re_true = []
for h in h_space:
    pole = eig(expm(Ac * h))[0][3]
    # print(pole)
    re.append(np.real(pole))
    true_pole = np.exp(Ac_poles[2] * h)
    re_true.append(true_pole)
    # print(re)
    # print(im)
plt.subplot(1, 2, 1)
plt.title("Location of pole " + str(1))
plt.plot(h_space, re, label=r'$e^{A_ch}$')
plt.plot(h_space, re_true, label=r'$e^{s_ih}$')
plt.xlabel("h")
plt.ylabel("Pole location")
plt.legend()

re = []
re_true = []
for h in h_space:
    pole = eig(expm(Ac * h))[0][1]
    # print(pole)
    re.append(np.real(pole))
    true_pole = np.exp(Ac_poles[3] * h)
    re_true.append(true_pole)
    # print(re)
    # print(im)
plt.subplot(1, 2, 2)
plt.title("Location of pole " + str(2))
plt.plot(h_space, re, label=r'$e^{A_ch}$')
plt.plot(h_space, re_true, label=r'$e^{s_ih}$')
plt.xlabel("h")
plt.ylabel("Pole location")
plt.legend()
plt.tight_layout()
# plt.savefig("sim_poles.pdf")
plt.show()
