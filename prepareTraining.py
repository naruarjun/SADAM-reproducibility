import csv 
import torch
import torch.nn
import torchvision 
from torchvision import datasets, transforms 
import model as M

def get_dataset(name, batchsize = 64) : 
  """ Function to import datasets to be used for training """ 
  assert name in ["mnist","cifar10", "cifar100"], "Improper dataset name given"
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
  loaderDict = {
      "mnist" : datasets.MNIST , 
      "cifar10" : datasets.CIFAR10 , 
      "cifar100" : datasets.CIFAR100 
  } 
  datasetDist = {
      "mnist" : (28, 10, 1) , ## 28*28 
      "cifar10" : (32, 10, 3) , ## 32*32  
      "cifar100" : (32, 100, 3) # 32*32, 100 classes
  } 
  trainset = loaderDict[name](root = './data' + name, train=True, transform=transform, download=False)
  testset = loaderDict[name](root = './data' + name, train=False, transform=transform, download=False)
  
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)  
  testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False)

  input_size, num_classes, channels = datasetDist[name]
  return trainloader, testloader, input_size, num_classes, channels

def get_model(name, input_size, num_classes, channels) : 
  assert name in ["logistic","nn"], "Improper model type given"
  if name == "logistic" : 
    return M.LogisticRegression(input_size, num_classes)
  else : 
    return M.Layer4NN(input_size, num_classes, channels)

def get_loss(losstype) : 
  return torch.nn.CrossEntropyLoss() 

def get_optimizer(params, name) : 
  assert name in ["adam", "adamnc", "sadam", "amsgrad", "scrms", "scadagrad", "ogd"], "Unknown Optimization"
  optimizers = {
      "adam" : torch.optim.Adam(params),
      "amsgrad" : torch.optim.Adam(params, amsgrad = True), 
      "scrms" : "ToBeDone", 
      "scadagrad" : "ToBeDone", 
      "adamnc" : "ToBeDone",
      "ogd" : "ToBeDone",
      "sadam" : "ToBeDone" 
  } 
  ## Sadam Optimizer to be given here later 
  return optimizers[name]

def write_details(loss, acc, filename) : 
    with open(filename, 'a') as output : 
        writer = csv.writer(output, delimiter = ",", lineterminator = '\n')
        writer.writerow(["Epochs", "Loss", "Accuracy"])
        for idx in range(loss) : 
            writer.writerow([idx, loss[idx], acc[idx]])
