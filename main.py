import matplotlib.pyplot as plt
import numpy as np
from ddeint import ddeint

beta = 6.2e-5
k = 4.0
p = 1.0
c = 9.4
delta = 2.4e-1
delta_e = 1.9
K_delta_e = 4.3e2
xi = 2.6e4
K_E = 8.1e5
eta = 2.5e-7
tau_e = 3.6
d_E = 1.0
zeta = 2.2e-1
tau_M = 3.5

def model(Y, t):
    T, I_1, I_2, V, E, E_M = Y(t)

    dT = -beta * T * V
    dI_1 = beta * T * V - k * I_1
    dI_2 = k * I_1 - delta * I_2 - ((delta_e * E) / (K_delta_e + I_2)) * I_2
    dV = p * I_2 - c * V
    dE = (xi / (K_E + E)) * I_2 + eta * E * Y(t - tau_e)[2] - d_E * E
    dE_M = zeta * Y(t - tau_M)[4]

    return [dT, dI_1, dI_2, dV, dE, dE_M]

if __name__ == '__main__':
    delta_t = 20000
    t = np.linspace(0.0, 12.0, delta_t)
    y_0 = lambda t: (1.0e7, 75, 0, 0, 0, 0)

    sol = ddeint(model, y_0, t)
    #plt.plot(np.linspace(0, 12, delta_t), sol[:, 3])
    plt.plot(np.linspace(0, 12, delta_t), np.log10(4.2e5 + np.add(sol[:, 4], sol[:, 5])))
    plt.ylim(5.5, 6.5)
    plt.show()