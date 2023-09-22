import matplotlib.pyplot as plt
import numpy as np
from ddeint import ddeint


class DDEViralKineticsModel:
    def __init__(self, beta=6.2e-5, k=4.0, p=1.0, c=9.4, delta=2.4e-1, delta_e=1.9,
                 k_delta_e=4.3e2, xi=2.6e4, k_e=8.1e5, eta=2.5e-7, tau_e=3.6,
                 d_e=1.0, zeta=2.2e-1, tau_m=3.5, initial_cd8=4.2e5):
        """
        Initializes the Viral Kinetics DDE model as described in the following paper.
        System values default to the "best-fit" parameters determined there.
            Margaret A Myers, Amanda P Smith, Lindey C Lane, David J Moquin, Rosemary Aogo, Stacie Woolard, Paul Thomas,
            Peter Vogel, Amber M Smith (2021) Dynamically linking influenza virus infection kinetics, lung injury,
            inflammation, and disease severity eLife 10:e68864
            https://doi.org/10.7554/eLife.68864
        :param beta: Virus Infectivity, 6.2 x 10^-5
        :param k: Eclipse phase transition, 4.0
        :param p: Virus Production, 1.0
        :param c: Virus Clearance, 9.4
        :param delta: Infected Cell Clearance, 2.4 x 10^-1
        :param delta_e: Infected Cell Clearance by CD8e, 1.9
        :param k_delta_e: Half-saturation constant for delta_e, 4.3 x 10^2
        :param xi: CD8 Infiltration, 2.6 x 10^4
        :param k_e: Half-saturation constant for CD8e, 8.1 x 10^5
        :param eta: CD8e expansion, 2.5 x 10^-7
        :param tau_e: Delay in CD8e expansion, 3.6
        :param d_e: CD8e clearance, 1.0
        :param zeta: CD8m generation, 2.2 x 10^-1
        :param tau_m: Delay in CD8m generation, 3.5
        """

        self._beta = beta
        self._k = k
        self._p = p
        self._c = c
        self._delta = delta
        self._delta_e = delta_e
        self._k_delta_e = k_delta_e
        self._xi = xi
        self._k_e = k_e
        self._eta = eta
        self._tau_e = tau_e
        self._d_e = d_e
        self._zeta = zeta
        self._tau_m = tau_m
        self._initial_cde8 = initial_cd8

        # _model_equation_builder without the actual parameters, to be used by ddeint, which has strict parameter needs
        self.model_equation = lambda y, t: self._model_equation_builder(y, t, self._beta, self._k, self._p, self._c,
                                                                        self._delta, self._delta_e, self._k_delta_e,
                                                                        self._xi, self._k_e, self._eta, self._tau_e,
                                                                        self._d_e, self._zeta, self._tau_m)

    @staticmethod
    def _model_equation_builder(y, t, beta, k, p, c, delta, delta_e,
                                k_delta_e, xi, k_e, eta, tau_e,
                                d_e, zeta, tau_m):
        """
        A function to "build" the actual model_equation used by ddeint
        Parameters are identical to ddeint's required parameters and the object's instance variables
        Not to be used directly.
        """

        T, I_1, I_2, V, E, E_M = y(t)

        dT = -beta * T * V
        dI_1 = beta * T * V - k * I_1
        dI_2 = k * I_1 - delta * I_2 - ((delta_e * E) / (k_delta_e + I_2)) * I_2
        dV = p * I_2 - c * V
        dE = (xi / (k_e + E)) * I_2 + eta * E * y(t - tau_e)[2] - d_e * E
        dE_M = zeta * y(t - tau_m)[4]

        return [dT, dI_1, dI_2, dV, dE, dE_M]

    def solve(self, delta_t, t_1, t_2, y_0):
        """
        Solves the DDE system iteratively and outputs the result
        :param delta_t: The amount of time per "step", in days
        :param t_1: The starting time, in days
        :param t_2: The ending time, in days
        :param y_0: The initial conditions for the system
        :return: A list, consisting of tuples "(T, I_1, I_2, V, E, E_M)" for each time step
        """
        assert t_2 > t_1, "t_2 must be greater than t_1"

        steps = int((t_2 - t_1) / delta_t)
        t = np.linspace(t_1, t_2, steps)
        return ddeint(self.model_equation, lambda x: y_0, t)

    def create_graph(self, graph_type, location, delta_t, t_1, t_2, y_0, as_log):
        solution = self.solve(delta_t, t_1, t_2, y_0)

        if as_log:
            f = lambda x: np.log10(x)
        else:
            f = lambda x: x

        steps = int((t_2 - t_1) / delta_t)
        t = np.linspace(t_1, t_2, steps)

        figure = plt.figure()
        figure.gca().grid()
        figure.gca().set_xlabel("Time (days)", fontsize=10)

        match graph_type:
            case 'T':
                figure.gca().set_title("Target Cells Over Time", fontsize=20)
                figure.gca().set_ylabel("Total Target Cells", fontsize=10)
                figure.gca().plot(t, f(solution[:, 0]))
                figure.gca().set_ylim(bottom=0)
            case 'I_1':
                figure.gca().set_title("Pre-Infected Cells Over Time", fontsize=20)
                figure.gca().set_ylabel("Total Pre-Infected Cells", fontsize=10)
                figure.gca().plot(t, f(solution[:, 1]))
                figure.gca().set_ylim(bottom=0)
            case 'I_2':
                figure.gca().set_title("Infected Cells Over Time", fontsize=20)
                figure.gca().set_ylabel("Total Infected Cells", fontsize=10)
                figure.gca().plot(t, f(solution[:, 2]))
                figure.gca().set_ylim(bottom=0)
            case 'V':
                figure.gca().set_title("Virus Over Time", fontsize=20)
                figure.gca().set_ylabel("Total Virus", fontsize=10)
                figure.gca().plot(t, f(solution[:, 3]))
                figure.gca().set_ylim(bottom=0)
            case 'CDE8':
                figure.gca().set_title("CDE8 Cells Over Time", fontsize=20)
                figure.gca().set_ylabel("Total CDE8 Cells", fontsize=10)
                figure.gca().plot(t, f(self._initial_cde8 + np.add(solution[:, 4], solution[:, 5])))
                figure.gca().set_ylim(bottom=f(self._initial_cde8))
            case 'E':
                figure.gca().set_title("CDE8e Cells Over Time", fontsize=20)
                figure.gca().set_ylabel("Total CDE8e Cells", fontsize=10)
                figure.gca().plot(t, f(solution[:, 4]))
                figure.gca().set_ylim(bottom=0)
            case 'E_M':
                figure.gca().set_title("CDE8m Cells Over Time", fontsize=20)
                figure.gca().set_ylabel("Total CDE8m Cells", fontsize=10)
                figure.gca().plot(t, f(solution[:, 5]))
                figure.gca().set_ylim(bottom=0)
            case default:
                assert False, "Invalid Graph Type"

        figure.savefig(location)
        figure.show()

if __name__ == '__main__':
    model = DDEViralKineticsModel()
    model.create_graph('T', "graphs/Target_log10.png", 0.001, 0, 12, (1.0e7, 75, 0, 0, 0, 0,), True)
    model.create_graph('I_1', "graphs/Pre-Infected_log10.png", 0.001, 0, 12, (1.0e7, 75, 0, 0, 0, 0,), True)
    model.create_graph('I_2', "graphs/Infected_log10.png", 0.001, 0, 12, (1.0e7, 75, 0, 0, 0, 0,), True)
    model.create_graph('V', "graphs/Virus_log10.png", 0.001, 0, 12, (1.0e7, 75, 0, 0, 0, 0,), True)
    model.create_graph('CDE8', "graphs/CDE8_log10.png", 0.001, 0, 12, (1.0e7, 75, 0, 0, 0, 0, ), True)
    model.create_graph('E', "graphs/CDE8e_log10.png", 0.001, 0, 12, (1.0e7, 75, 0, 0, 0, 0,), True)
    model.create_graph('E_M', "graphs/CDE8m_log10.png", 0.001, 0, 12, (1.0e7, 75, 0, 0, 0, 0,), True)