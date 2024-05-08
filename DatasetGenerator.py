"""
A generator for ViralKineticsDDE datasets. For use in training ViralKineticsDNN. Read the docs and generate using simple commandline args.

All variables are defined the same as the paper below:
    Margaret A Myers, Amanda P Smith, Lindey C Lane, David J Moquin, Rosemary Aogo, Stacie Woolard, Paul Thomas,
    Peter Vogel, Amber M Smith (2021) Dynamically linking influenza virus infection kinetics, lung injury,
    inflammation, and disease severity eLife 10:e68864
    https://doi.org/10.7554/eLife.68864
"""

import numpy as np
import pandas as pd
import random
import argparse

from ViralKineticsDDE import DDEViralKineticsModel
from tqdm import tqdm
from itertools import product
from pathlib import Path

def _generate_within_confidence(confidence_interval: list, num_choices):
    """
    Creates num_choices variables within the given confidence interval, and returns them as a list

    :param confidence_interval: A range of values as a list, particularly the confidence intervals defined above. 
    :param num_choices: the number of values to generate
    :return: A list of the generated values in random order.
    """

    assert num_choices > 0, "number of choices needs to be >= 1"

    generations = []
    for _ in range(num_choices):
        generations.append(random.choice(confidence_interval))
    return generations

def _generate_file_name(variable_elements, num_choices, solving_timestep, reference_timestep, t_0, t_n, y_0):
    """
    Builds the file name based on the desired elements.

    :param variable_elements: A list of the elements which are non-default. Each variable will be added the filename. Empty lists will have "none" instead of a list of the variables in the filename
    :param num_choices: For each variable element, how many non-default variables are in the dataset. Will only be included in the filename if variable_elements is not empty
    :param solving_timestep: The discretized timestep for solving the DDE system defined in the above paper, in days. Always included in the filename.
    :param reference_timestep: The discretized timestep for solving the DDE system defined in the above paper, in days. Always included in the filename.
    :param t_0: the starting time, in days. Always included in the filename.
    :param t_n: the ending time, in days. Always included in the filename.
    :param y_0: the initial value for the dde system. Always included in the filename.
    """

    file_name = "viral_kinetics_"
    if len(variable_elements) == 0:
        return file_name + "none_" + str(solving_timestep) + "_" + str(reference_timestep) + "_" + str(t_0) + "_" + str(t_n) + "_" + str(y_0) + ".csv"
    else:
        for element in variable_elements:
            file_name = file_name + element + "_"
        return file_name + str(num_choices) + "_" + str(solving_timestep) + "_" + str(reference_timestep) + "_" + str(t_0) + "_" + str(t_n) + "_" + str(y_0) + ".csv"

def generate_viral_kinetics_dataset(variable_elements: list, num_choices: int, solving_timestep: float, reference_timestep: float, t_0: float, t_n: float, y_0: list):
    """
    Generates and saves a dataset with the given parameters in a "data" folder. Datapoints take the form (xTarget, xPre-Infected, xInfected, xVirus, xCDE8e, xCD8m, yTarget, yPre-Infected, yInfected, yVirus, yCDE8e, yCD8m)
    Unless variable_elements is not empty, only the ideal curve defined in the above paper will be generated. All permutations of variable elements generate a seperate solution curve. Each curve is part of the dataset.

    :param variable_elements: A list of the system variables that will be randomly generated in the confidence intervals defined above. Empty lists will generate based on the ideal values decided in the above referenced paper
    :param num_choices: For each variable element, how many non-default variables are in the dataset. Later, the dataset is re-generated multiple times with all the combinations of the variables. Typically, just 1.
    :param solving_timestep: The discretized timestep for solving the DDE system defined in the above paper, in days. Smaller values create more accurate solutions
    :param reference_timestep: The discretized timestep for solving the DDE system defined in the above paper, in days I.e, a value of 1 means we are targeting datapoints 1 day away
    :param t_0: the starting time, in days
    :param t_n: the ending time, in days
    :param y_0: the initial value for the dde system
    """

    variable_elements = [element.lower() for element in variable_elements]

    if len(variable_elements) > 0:
        assert num_choices > 0, "non-empty variable elements requires num_choices >= 1"

    # The "ideal" values for each system variable, as determined the above paper, are included by default. Will be REPLACED if variable elements is not empty
    element_value_lists = {
        'beta' : [6.2e-5],
        'k' : [4.0],
        'p' : [1.0],
        'c' : [9.4],
        'delta' : [2.4e-1],
        'delta_e' : [1.9],
        'k_delta_e' : [4.3e2],
        'xi' : [2.6e4],
        'k_e' : [8.1e5],
        'eta' : [2.5e-7],
        'tau_e' : [3.6],
        'd_e' : [1.0],
        'zeta' : [2.2e-1],
        'tau_m' : [3.5]
    }         

    # Generating random variables for each desired variable element. Will only be generated within the confidence intervals defined the above paper.
    actual_variable_elements = []

    for element in variable_elements:
        actual_variable_elements.append(element)
        element_value_lists[element] = _generate_within_confidence(CONFIDENCE_INTERVALS[element], num_choices)

    # Creates a list of systems. Will be all combinations of systems as determined by the desired variable_elements.
    systems = []
    for system_combination in tqdm(product(element_value_lists['beta'], element_value_lists['k'], element_value_lists['p'], element_value_lists['c'], element_value_lists['delta'], element_value_lists['delta_e'],
                                           element_value_lists['k_delta_e'], element_value_lists['xi'], element_value_lists['k_e'], element_value_lists['eta'], element_value_lists['tau_e'], element_value_lists['d_e'],
                                           element_value_lists['zeta'], element_value_lists['tau_m']), desc="Generating Combinations"):
        systems.append(system_combination)

    # Computes the solution for each system in 'systems'. Iterates through the solution and generates datapoints in the form defined above.
    datapoints = []
    for system in tqdm(systems, desc="Computing Solutions"):
        model = DDEViralKineticsModel(*system)
        solution = model.solve(solving_timestep, t_0, t_n, y_0)

        counter = 0
        # Important note, we LOSE datapoints for larger reference timesteps. I.e, a reference timestep of 1 day loses 1 days worth of datapoints. 
        while counter < len(solution) - int(reference_timestep / solving_timestep):
            x = solution[counter]
            y = solution[counter + int(reference_timestep / solving_timestep)]
            counter += 1
            datapoints.append(np.append(x, y))

    print("Moving Data to Dataframe")
    df = pd.DataFrame(datapoints)

    print("Saving CSV")
    path = Path("./data")
    path.mkdir(exist_ok=True)
    df.to_csv(path / _generate_file_name(actual_variable_elements, num_choices, solving_timestep, reference_timestep, t_0, t_n, y_0),
              header=["xTarget", "xPre-Infected", "xInfected", "xVirus", "xCDE8e", "xCD8m",
                      "yTarget", "yPre-Infected", "yInfected", "yVirus", "yCDE8e", "yCD8m"],
              index=False)
    print("Finished!")

if __name__ == "__main__":
    # Commandline parsing. The arguments correspond to the parameters of generate_viral_kinetics_dataset. If you wish to use a non-default argument, enter the value with the corresponding --arg.
    parser = argparse.ArgumentParser()
    parser.add_argument('--variables', nargs='*')
    parser.add_argument('--numchoices', nargs='?')
    parser.add_argument('--solvingtimestep', nargs='?')
    parser.add_argument('--referencetimestep', nargs='?')
    parser.add_argument('--t0', nargs='?')
    parser.add_argument('--tn', nargs='?')
    parser.add_argument('--y0', nargs='*')

    args = parser.parse_args()

    # The default arguments. I.e, the "ideal" curve in the above paper referencing 1 day away.
    variables = []
    num_choices = 1
    solving_timestep = 0.001
    reference_timestep = 1
    t0 = 0
    tn = 12
    y0 = [1.0e7, 75, 0, 0, 0, 0]

    if args.variables is not None:
        variables = args.variables
    if args.numchoices is not None:
        num_choices = int(args.num_choices)
    if args.solvingtimestep is not None:
        solving_timestep = float(args.solvingtimestep)
    if args.referencetimestep is not None:
        reference_timestep = float(args.referencetimestep)
    if args.t0 is not None:
        t0 = int(args.t0)
    if args.tn is not None:
        tn = int(args.tn)
    if args.y0 is not None:
        tn = args.y0

    """
    A set of ranges for the various system variables defined for the viral kinetics DDE system described above

    They follow the 95% confidence intervals described in Table 1 of the results
    CONFIDENCE_CHOICES is an integer discretization variable. When choosing a random value in these ranges, you have CONFIDENCE_CHOICES choices, equal length apart.
    """
    CONFIDENCE_CHOICES = 1000
    CONFIDENCE_INTERVALS = {
        'beta' : np.linspace(5.3e-6, 1.0e-4, CONFIDENCE_CHOICES),
        'k' : np.linspace(4.0, 6.0, CONFIDENCE_CHOICES),
        'p' : np.linspace(5.8e-1, 1.1e2, CONFIDENCE_CHOICES),
        'c' : np.linspace(5.6, 9.5e2, CONFIDENCE_CHOICES),
        'delta' : np.linspace(1.0e-1, 6.6e-1, CONFIDENCE_CHOICES),
        'delta_e' : np.linspace(3.3e-1, 2.0, CONFIDENCE_CHOICES),
        'k_delta_e' : np.linspace(1.0e2, 2.0e5, CONFIDENCE_CHOICES),
        'xi' : np.linspace(1.3e2, 8.7e4, CONFIDENCE_CHOICES),
        'k_e' : np.linspace(1.0e3, 1.0e6, CONFIDENCE_CHOICES),
        'eta' : np.linspace(1.6e-8, 6.7e-7, CONFIDENCE_CHOICES),
        'tau_e' : np.linspace(2.1, 5.9, CONFIDENCE_CHOICES),
        'd_e' : np.linspace(5.1e-2, 2.0, CONFIDENCE_CHOICES),
        'zeta' : np.linspace(1.0e-2, 9.4e-1, CONFIDENCE_CHOICES),
        'tau_m' : np.linspace(3.0, 4.0, CONFIDENCE_CHOICES)
    }

    generate_viral_kinetics_dataset(variables, num_choices, solving_timestep, reference_timestep, t0, tn, y0)