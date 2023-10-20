import NNModel
import matplotlib.pyplot as plt
import torch


def generate_graph(graph_type, iterations, y_0, checkpoint_path):
    figure = plt.figure()
    figure.gca().grid()

    match graph_type:
        case 'T':
            figure.gca().set_title("Target Cells Over Time", fontsize=20)
            figure.gca().set_ylabel("Total Target Cells", fontsize=10)
            index = 0
        case 'I_1':
            figure.gca().set_title("Pre-Infected Cells Over Time", fontsize=20)
            figure.gca().set_ylabel("Total Pre-Infected Cells", fontsize=10)
            index = 1
        case 'I_2':
            figure.gca().set_title("Infected Cells Over Time", fontsize=20)
            figure.gca().set_ylabel("Total Infected Cells", fontsize=10)
            index = 2
        case 'V':
            figure.gca().set_title("Virus Over Time", fontsize=20)
            figure.gca().set_ylabel("Total Virus", fontsize=10)
            index = 3
        case 'E':
            figure.gca().set_title("CDE8e Cells Over Time", fontsize=20)
            figure.gca().set_ylabel("Total CDE8e Cells", fontsize=10)
            index = 4
        case 'E_M':
            figure.gca().set_title("CDE8m Cells Over Time", fontsize=20)
            figure.gca().set_ylabel("Total CDE8m Cells", fontsize=10)
            index = 5
        case default:
            assert False, "Invalid Graph Type"

    model = NNModel.ViralKineticsDNN.load_from_checkpoint(checkpoint_path)
    x = y_0
    x_values = []
    y_values = []

    for i in range(iterations):
        x_values.append(i)
        y_values.append(x[index])
        y = model(torch.Tensor(x).double().cuda())
        x = y.tolist()

    figure.gca().plot(x_values, y_values)
    figure.show()


if __name__ == '__main__':
    generate_graph('E', 1200, (1.0e7, 75, 0, 0, 0, 0,),
                   "lightning_logs/0.01 to 0.01 Model/version_1/checkpoints/sample-mnist-epoch=3684-testing_loss=13940657859.90796.ckpt")