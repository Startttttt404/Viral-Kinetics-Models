"""
Allows for the generation of heatmaps based on the results of ViralKineticsDNN experiments.
Must have the latest version of seaborn installed!! or else your heatmaps will be missing values in the boxes
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pathlib import Path
from tqdm import tqdm

def generate_heatmap(pivot1, pivot2, const1, const1_value, const2, const2_value, third_dim):
    """
    Generates and saves a heatmap based on the given parameters to /graphs/heatmaps
    Our result files have 4 model parameters: input_features, output_feature, data_usage, and num_outputs. We will generate 2d heatmaps, so we must choose 2 which will pivot and 2 which will stay constant.
    Our third dimension should be average_final_validation_loss, average_final_validation_accuracy, average_testing_loss, or average_testing_accuracy, depending on what you want to look at.

    :param pivot1: a str, represents the 1st parameter to pivot. Will be the y-axis. Must be input_features, output_feature, data_usage, or num_outputs
    :param pivot2: a str, represents the 2nd parameter to pivot. Will be the x-axis. Must be input_features, output_feature, data_usage, or num_outputs
    :param const1: a str, represents the 1st constant parameter. Must be input_features, output_feature, data_usage, or num_outputs
    :param const1_value: type is dependent on const1, but needs to be the value of the constant parameter.
    :param const2: a str, represents the 1st constant parameter. Must be input_features, output_feature, data_usage, or num_outputs
    :param const2_value: type is dependent on const2, but needs to be the value of the constant parameter.
    :param third_dim: a str, represents which value to place in each tile of the heatmap. Must be average_final_validation_loss, average_final_validation_accuracy, average_testing_loss, or average_testing_accuracy
    """

    assert pivot1 in ["input_features", "data_usage", "num_outputs", "output_feature"], "pivot1 must be input_features, output_feature, data_usage, or num_outputs"
    assert pivot2 in ["input_features", "data_usage", "num_outputs", "output_feature"], "pivot2 must be input_features, output_feature, data_usage, or num_outputs"
    assert const1 in ["input_features", "data_usage", "num_outputs", "output_feature"], "const1 must be input_features, output_feature, data_usage, or num_outputs"
    assert const1 in ["input_features", "data_usage", "num_outputs", "output_feature"], "const2 must be input_features, output_feature, data_usage, or num_outputs"
    assert third_dim in ['average_final_validation_loss', 'average_final_validation_accuracy', 'average_testing_loss', 'average_testing_accuracy'], "third_dim must be average_final_validation_loss, average_final_validation_accuracy, average_testing_loss, or average_testing_accuracy"

    data = pd.read_csv("results/results.csv", index_col=0)
    
    # (0, 1, 2, 3, 4, 5) is great and all, but the actual variable names is more readable
    data["input_features"] = data["input_features"].apply(input_features_to_names)

    # Getting the correct rows for const1 and 2, then deleting the const1 and 2 columns
    data = data.loc[data[const1] == const1_value]
    data = data.loc[data[const2] == const2_value]
    data = data.drop(const1, axis=1)
    data = data.drop(const2, axis=1)

    data = data.pivot_table(index=pivot1, columns=pivot2, values=third_dim)
    sns.heatmap(data, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.suptitle(pivot1 + " vs. " + pivot2 + " on " + third_dim)
    plt.title(const1 + ": " + str(const1_value) + ", " + const2 + ": " + str(const2_value))
    plt.tight_layout()

    graphs_path = Path("./graphs")
    graphs_path.mkdir(exist_ok=True)
    heatmaps_path = graphs_path / "heatmaps"
    heatmaps_path.mkdir(exist_ok=True)

    plt.savefig(heatmaps_path / str(pivot1 + "_vs_" + pivot2 + "_on_" + third_dim + const1 + str(const1_value) + const2 + str(const2_value) + ".png"))
    plt.close()

def input_features_to_names(features):
    """
    Converts and returns a more clear str from a set of input_features. Since I've only ever removed 1 variable from the set, that is all that is available.
    If you want to create heatmaps for other values of input_features, you must add the cases yourself.

    :param features: the set of input features, as a list. More rigorously defined in ViralKineticsDNN's parameters.
    
    :returns: a str which is a more "human" representation of features
    """
    match features:
        case "(0, 1, 2, 3, 4, 5)":
            return "all features"
        case "(0, 1, 2, 3, 4)":
            return "no E_M"
        case "(0, 1, 2, 3, 5)":
            return "no E"
        case "(0, 1, 2, 4, 5)":
            return "no V"
        case "(0, 1, 3, 4, 5)":
            return "no I_2"
        case "(0, 2, 3, 4, 5)":
            return "no I_1"
        case "(1, 2, 3, 4, 5)":
            return "no T"

if __name__ == "__main__":
    """
    A quick example of automating a bunch of heatmaps.
    We are keeping num_outputs and output_features constant for each heatmap, and pivoting along input_features and data_usage
    """

    for num_outputs in tqdm([4, 8, 16], desc="Generating Heatmap for each num_output in Range", leave=False):
        for output_feature in tqdm([2, 3, 4, 5], desc="Generating Heatmap for each output_feature in Range", leave=False):
            for third_dim in tqdm(["average_testing_accuracy", "average_testing_loss"], desc="Generating Heatmap for both average_testing_accuracy and average_testing_loss", leave=False):
                generate_heatmap("input_features", "data_usage", "num_outputs", num_outputs, "output_feature", output_feature, third_dim=third_dim)

    print("Heatmaps Generated!")