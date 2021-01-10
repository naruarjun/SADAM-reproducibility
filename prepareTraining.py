import csv 
import torch
import wandb 
import numpy as np
import torch.nn
import torchvision 
from tqdm import tqdm 
from torchvision import datasets, transforms 
import model as M
import custom_optimizers as OP

def getNumCorrect(correct, outputs, labels) : 
    ## For computing Accuracy 
    _, predicted = torch.max(outputs.data, 1)
    labelsTemp = labels.to("cpu")  
    predicted = predicted.to("cpu") 
    return correct + (predicted == labelsTemp).sum().item()

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
  try : 
      trainset = loaderDict[name](root = './data' + name, train=True, transform=transform, download=False)
      testset = loaderDict[name](root = './data' + name, train=False, transform=transform, download=False)
  except Exception as e : 
      trainset = loaderDict[name](root = './data' + name, train=True, transform=transform, download=True)
      testset = loaderDict[name](root = './data' + name, train=False, transform=transform, download=True)
  
  torch.backends.cudnn.deterministic = True
  torch.manual_seed(1)
  torch.cuda.manual_seed(1)
  np.random.seed(1)

  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)  
  testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False)

  input_size, num_classes, channels = datasetDist[name]
  return trainloader, testloader, input_size, num_classes, channels, len(trainset)

def get_model(name, input_size, num_classes, channels) : 
  assert name in ["logistic","nn"], "Improper model type given"
  if name == "logistic" : 
    return M.LogisticRegression(input_size, num_classes, channels)
  else : 
    return M.Layer4NN(input_size, num_classes, channels)

def get_loss(losstype) : 
  return torch.nn.CrossEntropyLoss() 

def get_optimizer(params, name, lr, convex = False) : 
  assert name in ["adam", "adamnc", "sadam", "amsgrad", "scrms", "scadagrad", "ogd"], "Unknown Optimization"
  
  optimizers = {
      "adam" : torch.optim.Adam(params, lr=lr),
      "amsgrad" : torch.optim.Adam(params, lr=lr, amsgrad = True), 
      "scrms" : OP.SC_RMSprop(params, lr=lr, convex=convex), 
      "scadagrad" : OP.SC_Adagrad(params, lr=lr, convex=convex), 
      "ogd" : OP.SC_SGD(params, convex, lr=lr),
      "sadam" : OP.SAdam(params, lr=lr)
  } 
  return optimizers[name]

def train_model(model, lossfn, device, epochs, optimizer, train_loader, test_loader = "", reshape = True) :

    logging_dict = {} 
    for epoch in tqdm(range(int(epochs))) :
        correct, total, epoch_loss = 0, 0, 0.0

        ## Training Epoch 
        for data in train_loader :
            images, labels = data[0].to(device), data[1].to(device)
            if reshape : 
                images = images.reshape((images.shape[0], -1))
            model.eval() 
            optimizer.zero_grad()
            outputs = model(images)
            model.train()

            ## For computing Accuracy 
            total += labels.size(0) ## batch size added, at each step
            correct = getNumCorrect(correct, outputs, labels) 

            ## Compute Loss 
            loss = lossfn(outputs, labels)
            loss.backward()
            epoch_loss += loss.item()
            
            ## Optimizer Step and scheduler step 
            optimizer.step()

        train_accuracy = 100*correct/total
        logging_dict["TrainLoss"] = epoch_loss
        logging_dict["TrainAccuracy"] = 100 * correct/total

        ## Testing Epoch 
        if test_loader : 
            model.eval() 
            correct, total, testloss = 0, 0, 0.0
            for imagesT, labelsT in test_loader :
                ## Get Model output 
                imagesT, labelsT = imagesT.to(device), labelsT.to(device)
                outputsT = model(imagesT)

                ## For calculating metrics to log 
                total += labelsT.size(0)
                correct = getNumCorrect(correct, outputsT, labelsT) 
                lossTest = lossfn(outputsT, labelsT) 
                testloss += lossTest.item() 
            
            logging_dict["TestLoss"] = testloss  
            logging_dict["TestAccuracy"] = 100 * correct/total
            model.train() 

        wandb.log(logging_dict)
    return model

    
def regret_calculation(train_loader, modelT, optimizerT, lossfn, device, iterations, optim, convex, inpsize, classes, channels) :

    ## Params initialized 
    modelE, optimizerE, logging = [], [], {} 

    ## For reproducibility 
    torch.backends.cudnn.deterministic = True
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    running_regret_sum = []   
    modelT.eval()
    for lr in [0.01, 0.001, 0.0001] :
        ## Make the model & optimizer for each LR
        model = get_model("logistic", inpsize, classes, channels)
        model.to(device) 
        modelE.append(model) 
        optimizerE.append(get_optimizer(list(model.parameters()), optim, lr, convex))
        running_regret_sum.append(0)
    for itern, data in enumerate(train_loader) :
        images, labels = data[0].to(device), data[1].to(device)

        ## Initialize all evaluation optimizers to 0
        _ = [optim.zero_grad() for optim in optimizerE] 

        ## Get the output for all of the models 
        images = images.reshape((images.shape[0], -1))
        outputsT = modelT(images)
        outputsE = [m(images) for m in modelE]

        ## Get the loss from each of the models 
        lossT = lossfn(outputsT, labels)
        lossesE = [lossfn(output, labels) for output in outputsE] 

        ## BackProp only for the Evaluation models 
        _ = [loss.backward() for loss in lossesE]
        _ = [optimizer.step() for optimizer in optimizerE] 

        ## Insert the Regret Values in For logging 
        for idx, loss in enumerate(lossesE) : 
            logging["loss_"] = loss.item()
            # logging["normal"] = np.abs(loss.item() - lossT.item())
            running_regret_sum[idx] = running_regret_sum[idx] + (loss.item() - lossT.item())*images.size(0)
            logging["Regret_1e" + str(idx+2)] = running_regret_sum[idx]
        logging["DatasetProportion"] = (itern + 1)/iterations
        wandb.log(logging)
