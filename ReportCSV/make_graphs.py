import pandas as pd 
import matplotlib.pyplot as plt


def get_graph_from_csv(filename, names, xaxis):
	
	df = pd.read_csv(filename)
	dataset = df[xaxis]
	
	regrete2 = df.loc[:, df.columns.str.endswith('1e2')]
	regrete3 = df.loc[:, df.columns.str.endswith('1e3')]

	plot_columns = {}  
	for name in names : 
		optim_e3 = regrete3.loc[:, regrete3.columns.str.startswith(name)]
		print(" E3 ", optim_e3.iloc[-1])
		optim_e2 = regrete2.loc[:, regrete2.columns.str.startswith(name)]
		print(" E2 ", optim_e2.iloc[-1])

		max_e2 = float(optim_e2.iloc[-1])
		max_e3 = float(optim_e3.iloc[-1]) 

		if (max_e2 > max_e3) : ## Add the minimum of the 2 options 
			plot_columns[name] = optim_e3
		else : 
			plot_columns[name] = optim_e2
	
	return dataset, plot_columns 

def deep_get_graph_from_csv(filename, names, xaxis = 'Step', yaxis = 'Accuracy'):
	
	df = pd.read_csv(filename)
	dataset = df[xaxis]
	
	yaxis_metric = df.loc[:, df.columns.str.endswith(yaxis)]
	plot_columns = {}  
	for name in names : 
		yaxis_column = yaxis_metric.loc[:, yaxis_metric.columns.str.startswith(name)]
		print(yaxis, yaxis_column.iloc[-1])
		plot_columns[name] = yaxis_column
		
	return dataset, plot_columns 


def make_image(filename, plot_columns, names, dataset, ylabel_name, xlabel_name): 
	colour_list = ["purple", "blue", "yellow", "green", "red", "orange"]
	plt.figure() 
	for name,colour in zip(names, colour_list) : 
		plt.plot(dataset, plot_columns[name], color=colour, label = name)

	plt.title('', fontsize=14)
	plt.legend(loc="upper right")
	plt.xlabel(xlabel_name, fontsize=14)
	plt.ylabel(ylabel_name, fontsize=14)
	plt.grid(True)
	plt.savefig(filename[:-3] + 'png')
	plt.show()

# # For CIFAR 10 Regret
# names = ["Amsgrad", "SAdam", "Adam", "SC-RMSprop", "SC-Adagrad", "OGD"]
# # For Cifar 10, 4 layer 
# names = ["AmsGrad", "Sadam", "Adam", "SCRms", "SCAdagrad", "OGD"]
# # For Cifar 100, 4 layer 
names = ["AmsGrad", "Sadam", "Adam", "SC-RMS", "Sc-Adagrad", "OGD"]
# and ResNet  
names = ["AmsGrad", "Sadam", "Adam", "SC-RMS", "SC-Adagrad", "OGD"]
# For MNIST 
names = ["Amsgrad", "Sadam", "Adam", "SCRms", "SCadagrad", "OGD"]
## For Regret 
# dataset, plot_columns = get_graph_from_csv('mnist.csv', names, 'DatasetProportion')
# make_image(plot_columns, names, dataset, "Regret", "Dataset Proportion")

## For 4 layer and ResNet 
filename = 'mnistTrainLoss.csv'
dataset, plot_columns = deep_get_graph_from_csv(filename, names, 'Step', "Loss")
make_image(filename, plot_columns, names, dataset, "Train Loss", "Number of Epochs")
#make_image(plot_columns, names, dataset, "Train Loss", "Number of Epochs")
