import numpy as np
import pandas as pd
import random

from ODEModel import DDEViralKineticsModel
from tqdm import tqdm
from tqdm.contrib.itertools import product

CONFIDENCE_STEPS = 100
BETA_CONFIDENCE = np.linspace(5.3e-6, 1.0e-4, CONFIDENCE_STEPS)
K_CONFIDENCE = np.linspace(4.0, 6.0, CONFIDENCE_STEPS)
P_CONFIDENCE = np.linspace(5.8e-1, 1.1e2, CONFIDENCE_STEPS)
C_CONFIDENCE = np.linspace(5.6, 9.5e2, CONFIDENCE_STEPS)
DELTA_CONFIDENCE = np.linspace(1.0e-1, 6.6e-1, CONFIDENCE_STEPS)
DELTA_E_CONFIDENCE = np.linspace(3.3e-1, 2.0, CONFIDENCE_STEPS)
K_DELTA_E_CONFIDENCE = np.linspace(1.0e2, 2.0e5, CONFIDENCE_STEPS)
XI_CONFIDENCE = np.linspace(1.3e2, 8.7e4, CONFIDENCE_STEPS)
K_E_CONFIDENCE = np.linspace(1.0e3, 1.0e6, CONFIDENCE_STEPS)
ETA_CONFIDENCE = np.linspace(1.6e-8, 6.7e-7, CONFIDENCE_STEPS)
TAU_E_CONFIDENCE = np.linspace(2.1, 5.9, CONFIDENCE_STEPS)
D_E_CONFIDENCE = np.linspace(5.1e-2, 2.0, CONFIDENCE_STEPS)
ZETA_CONFIDENCE = np.linspace(1.0e-2, 9.4e-1, CONFIDENCE_STEPS)
TAU_M_CONFIDENCE = np.linspace(3.0, 4.0, CONFIDENCE_STEPS)

Y_0 = (1.0e7, 75, 0, 0, 0, 0,)


def generate_within_confidence(confidence_interval, num_choices):
    generations = []
    for i in range(0, num_choices):
        generations.append(random.choice(confidence_interval))
    return generations


def generate_file_name(variable_elements, num_choices, delta_t, t_0, t_n):
    file_name = "data/viral_kinetics_"

    if len(variable_elements) == 0:
        return file_name + "none_" + str(delta_t) + "_" + str(t_0) + "_" + str(t_n) + ".csv"
    else:
        for element in variable_elements:
            file_name = file_name + element + "_"
        return file_name + str(num_choices) + "_" + str(delta_t) + "_" + str(t_0) + "_" + str(t_n) + ".csv"


def generate_viral_kinetics_dataset(variable_elements, num_choices, delta_t, t_0, t_n):
    variable_elements = [element.lower() for element in variable_elements]

    beta = [6.2e-5]
    k = [4.0]
    p = [1.0]
    c = [9.4]
    delta = [2.4e-1]
    delta_e = [1.9]
    k_delta_e = [4.3e2]
    xi = [2.6e4]
    k_e = [8.1e5]
    eta = [2.5e-7]
    tau_e = [3.6]
    d_e = [1.0]
    zeta = [2.2e-1]
    tau_m = [3.5]

    actual_variable_elements = []
    if 'beta' in variable_elements:
        actual_variable_elements.append('beta')
        beta.extend(generate_within_confidence(BETA_CONFIDENCE, num_choices))
    if 'k' in variable_elements:
        actual_variable_elements.append('k')
        k.extend(generate_within_confidence(K_CONFIDENCE, num_choices))
    if 'p' in variable_elements:
        actual_variable_elements.append('p')
        p.extend(generate_within_confidence(P_CONFIDENCE, num_choices))
    if 'c' in variable_elements:
        actual_variable_elements.append('c')
        c.extend(generate_within_confidence(C_CONFIDENCE, num_choices))
    if 'delta' in variable_elements:
        actual_variable_elements.append('delta')
        delta.extend(generate_within_confidence(DELTA_CONFIDENCE, num_choices))
    if 'delta_e' in variable_elements:
        actual_variable_elements.append('delta_e')
        delta_e.extend(generate_within_confidence(DELTA_E_CONFIDENCE, num_choices))
    if 'k_delta_e' in variable_elements:
        actual_variable_elements.append('k_delta_e')
        k_delta_e.extend(generate_within_confidence(K_DELTA_E_CONFIDENCE, num_choices))
    if 'xi' in variable_elements:
        actual_variable_elements.append('xi')
        xi.extend(generate_within_confidence(XI_CONFIDENCE, num_choices))
    if 'k_e' in variable_elements:
        actual_variable_elements.append('k_e')
        k_e.extend(generate_within_confidence(K_E_CONFIDENCE, num_choices))
    if 'eta' in variable_elements:
        actual_variable_elements.append('eta')
        eta.extend(generate_within_confidence(ETA_CONFIDENCE, num_choices))
    if 'tau_e' in variable_elements:
        actual_variable_elements.append('tau_e')
        tau_e.extend(generate_within_confidence(TAU_E_CONFIDENCE, num_choices))
    if 'd_e' in variable_elements:
        actual_variable_elements.append('d_e')
        d_e.extend(generate_within_confidence(D_E_CONFIDENCE, num_choices))
    if 'zeta' in variable_elements:
        actual_variable_elements.append('zeta')
        zeta.extend(generate_within_confidence(ZETA_CONFIDENCE, num_choices))
    if 'tau_m' in variable_elements:
        actual_variable_elements.append('tau_m')
        tau_m.extend(generate_within_confidence(TAU_M_CONFIDENCE, num_choices))

    systems = []
    for system_permutation in product(beta, k, p, c, delta, delta_e, k_delta_e, xi, k_e, eta, tau_e, d_e, zeta, tau_m,
                                      desc="Computing Permutations"):
        systems.append(system_permutation)

    datapoints = []
    for system in tqdm(systems, desc="Computing Solutions"):
        model = DDEViralKineticsModel(*system)
        solution = model.solve(0.001, t_0, t_n, Y_0)

        counter = 0
        while counter < len(solution) - int(delta_t / 0.001):
            x = np.append(counter + int(delta_t / 0.001), solution[counter])
            y = solution[counter + int(delta_t / 0.001)]
            counter += 1
            datapoints.append(np.append(x, y))

    df = pd.DataFrame(datapoints)

    print("Saving CSV.")
    df.to_csv(generate_file_name(actual_variable_elements, num_choices, delta_t, t_0, t_n),
              header=["references", "xTarget", "xPre-Infected", "xInfected", "xVirus", "xCDE8e", "xCD8m",
                      "yTarget", "yPre-Infected", "yInfected", "yVirus", "yCDE8e", "yCD8m"],
              index=False)
    print("Finished!")


if __name__ == '__main__':
    # generate_viral_kinetics_dataset([], 1, 1, 0, 12)
    generate_viral_kinetics_dataset(['delta', 'd_e', 'eta'], 5, 1, 0, 12)

