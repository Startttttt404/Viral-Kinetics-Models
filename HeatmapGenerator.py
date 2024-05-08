import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_heatmap(pivot_1, pivot_2, const1, const1_parameter, const2, const2_parameter, third_dim="average_testing_accuracy"):
    data = pd.read_csv("results/results.csv", index_col=0)
    data["input_features"] = data["input_features"].apply(input_features_to_names)

    data = data.loc[data[const1] == const1_parameter]
    data = data.loc[data[const2] == const2_parameter]
    data = data.drop(const1, axis=1)
    data = data.drop(const2, axis=1)

    data = data.pivot_table(index=pivot_1, columns=pivot_2, values=third_dim)
    sns.heatmap(data, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.suptitle(pivot_1 + " vs. " + pivot_2 + " on " + third_dim)
    plt.title(const1 + ": " + str(const1_parameter) + ", " + const2 + ": " + str(const2_parameter))
    plt.tight_layout()
    plt.savefig("graphs/" + pivot_1 + "_vs_" + pivot_2 + "_on_" + third_dim + const1 + str(const1_parameter) + const2 + str(const2_parameter) + ".png")
    plt.show()
    plt.close()

def input_features_to_names(features):
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
    generate_heatmap("input_features", "data_usage", "num_buckets", 4, "output_feature", 2)
    generate_heatmap("input_features", "data_usage", "num_buckets", 8, "output_feature", 2)
    generate_heatmap("input_features", "data_usage", "num_buckets", 16, "output_feature", 2)
    generate_heatmap("input_features", "data_usage", "num_buckets", 4, "output_feature", 3)
    generate_heatmap("input_features", "data_usage", "num_buckets", 8, "output_feature", 3)
    generate_heatmap("input_features", "data_usage", "num_buckets", 16, "output_feature", 3)
    generate_heatmap("input_features", "data_usage", "num_buckets", 4, "output_feature", 4)
    generate_heatmap("input_features", "data_usage", "num_buckets", 8, "output_feature", 4)
    generate_heatmap("input_features", "data_usage", "num_buckets", 16, "output_feature", 4)
    generate_heatmap("input_features", "data_usage", "num_buckets", 4, "output_feature", 5)
    generate_heatmap("input_features", "data_usage", "num_buckets", 8, "output_feature", 5)
    generate_heatmap("input_features", "data_usage", "num_buckets", 16, "output_feature", 5)

    generate_heatmap("input_features", "data_usage", "num_buckets", 4, "output_feature", 2, third_dim="average_testing_loss")
    generate_heatmap("input_features", "data_usage", "num_buckets", 8, "output_feature", 2, third_dim="average_testing_loss")
    generate_heatmap("input_features", "data_usage", "num_buckets", 16, "output_feature", 2, third_dim="average_testing_loss")
    generate_heatmap("input_features", "data_usage", "num_buckets", 4, "output_feature", 3, third_dim="average_testing_loss")
    generate_heatmap("input_features", "data_usage", "num_buckets", 8, "output_feature", 3, third_dim="average_testing_loss")
    generate_heatmap("input_features", "data_usage", "num_buckets", 16, "output_feature", 3, third_dim="average_testing_loss")
    generate_heatmap("input_features", "data_usage", "num_buckets", 4, "output_feature", 4, third_dim="average_testing_loss")
    generate_heatmap("input_features", "data_usage", "num_buckets", 8, "output_feature", 4, third_dim="average_testing_loss")
    generate_heatmap("input_features", "data_usage", "num_buckets", 16, "output_feature", 4, third_dim="average_testing_loss")
    generate_heatmap("input_features", "data_usage", "num_buckets", 4, "output_feature", 5, third_dim="average_testing_loss")
    generate_heatmap("input_features", "data_usage", "num_buckets", 8, "output_feature", 5, third_dim="average_testing_loss")
    generate_heatmap("input_features", "data_usage", "num_buckets", 16, "output_feature", 5, third_dim="average_testing_loss")
