import argparse
import pandas as pd
import matplotlib.pyplot as plt


def get_graph_from_csv(filename, names, xaxis):

    df = pd.read_csv(filename)
    dataset = df[xaxis]

    regrete2 = df.loc[:, df.columns.str.endswith('1e2')]
    regrete3 = df.loc[:, df.columns.str.endswith('1e3')]

    plot_columns = {}
    for name in names:
        optim_e3 = regrete3.loc[:, regrete3.columns.str.startswith(name)]
        print(" E3 ", optim_e3.iloc[-1])
        optim_e2 = regrete2.loc[:, regrete2.columns.str.startswith(name)]
        print(" E2 ", optim_e2.iloc[-1])

        max_e2 = float(optim_e2.iloc[-1])
        max_e3 = float(optim_e3.iloc[-1])

        # Add the minimum of the 2 options
        if (max_e2 > max_e3):
            plot_columns[name] = optim_e3
        else:
            plot_columns[name] = optim_e2

    return dataset, plot_columns


def deep_get_graph_from_csv(filename, names, xaxis='Step', yaxis='Accuracy'):

    df = pd.read_csv(filename)
    dataset = df[xaxis]

    yaxis_metric = df.loc[:, df.columns.str.endswith(yaxis)]
    plot_columns = {}
    for name in names:
        yaxis_column = yaxis_metric.loc[:, yaxis_metric.columns.str.startswith(name)]
        print(yaxis, yaxis_column.iloc[-1])
        plot_columns[name] = yaxis_column

    return dataset, plot_columns


def make_image(filename, plot_columns, names, dataset, ylabel_name, xlabel_name):
    colour_list = ["purple", "blue", "yellow", "green", "red", "orange"]
    plt.figure()
    for name, colour in zip(names, colour_list):
        plt.plot(dataset, plot_columns[name], color=colour, label=name)

    plt.title('', fontsize=14)
    plt.legend(loc="upper right")
    plt.xlabel(xlabel_name, fontsize=14)
    plt.ylabel(ylabel_name, fontsize=14)
    plt.grid(True)
    plt.savefig(filename[:-3] + 'png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Denotes names of the runs, CSV files are downloaded from the Wandb page
    names = ["AmsGrad", "Sadam", "Adam", "SC-RMS", "Sc-Adagrad", "OGD"]

    parser.add_argument("--filename", help="File path", required=True, type=str)
    parser.add_argument("--xaxis_type", help="Step / Dataset Proportion", default="Step", type=str)
    parser.add_argument("--xaxis_name", help="Graph give the name of the X axis", default="Train Loss", type=str)
    parser.add_argument("--yaxis_name", help="Graph give the name of the Y axis", default="Step", type=str)
    parser.add_argument("--find_value", help = "Accuracy or Loss depending on the graph to be plotted", default="Accuracy", type=str)

    args = parser.parse_args()
    dataset, plot_columns = deep_get_graph_from_csv(args.filename, names, args.xaxis_type, args.find_value)
    make_image(filename, plot_columns, names, dataset, args.yaxis_name, args.xaxis_name)
