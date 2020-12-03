import os
import torch
import wandb 
import torch.cuda
import torch.nn
import torchvision 
from torchvision import datasets, transforms 
import argparse 
from tqdm import tqdm 
import prepareTraining as PT 

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def getNumCorrect(correct, outputs, labels) : 
    ## For computing Accuracy 
    _, predicted = torch.max(outputs.data, 1)
    labelsTemp = labels.to(safe_device)  
    predicted = predicted.to(safe_device) 
    return correct + (predicted == labelsTemp).sum().item() 

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--batch_size', type = int, default = 64)

parser.add_argument('--load_weights', type = str2bool, default = False, help = 'Load previous weights or not')
parser.add_argument('--load_file', type = str, default = 'dne.pth')
parser.add_argument('--details_file', type = str, default = 'LossAccuracy.csv', help = 'Save values for plotting later')

parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--decay', type = str2bool, default = False, help = 'Whether Decay for LR is to be done')
parser.add_argument('--model', type = str, default = 'nn')
parser.add_argument('--loss', type = str, default = 'entropy')
parser.add_argument('--dataset', type = str, default = 'mnist')
parser.add_argument('--convex', type = str2bool, default = False, help = 'Whether loss fn is convex or not')
parser.add_argument('--optimizer', type = str, default = 'adam')

args = parser.parse_args()
wandb.init(project = 'sadam') 

config = wandb.config          # Initialize config
config.batch_size =  args.batch_size         # input batch size for training (default: 64)
config.test_batch_size = args.batch_size    # input batch size for testing (default: 1000)
config.epochs = args.epochs             # number of epochs to train (default: 10)
config.log_interval = 10     # how many batches to wait before logging training status

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
safe_device = torch.device("cpu") 
lossfn = PT.get_loss(args.loss)
train_loader, test_loader, input_size, num_classes, channels = PT.get_dataset(args.dataset, args.batch_size) 

model = PT.get_model(args.model, input_size, num_classes, channels)
optimizer = PT.get_optimizer(list(model.parameters()), args.optimizer, args.lr, args.convex) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=25, gamma=0.7)
model.to(device) 

if args.load_weights : 
    model.load_state_dict(torch.load(args.load_file))
wandb.watch(model, log="all") 


for epoch in tqdm(range(int(args.epochs))) : 
    correct, total, trainloss = 0, 0, 0.0
    for iteration, data in enumerate(train_loader) :
        ## Get Model Output 
        images, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(images)

        ## For computing Accuracy 
        total += labels.size(0)
        correct = getNumCorrect(correct, outputs, labels) 

        ## Compute Loss 
        loss = lossfn(outputs, labels)
        trainloss += loss.item() 
        loss.backward()

        ## Optimizer Step and scheduler step 
        optimizer.step()
        if args.decay and epoch % 2 == 0 : 
            scheduler.step() 
    trainaccuracy = 100 * correct/total

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

    testaccuracy = 100 * correct/total
    wandb.log({"TrainLoss" : trainloss, "TestLoss" : testloss, "TrainAccuracy" : trainaccuracy, "TestAccuracy" :  testaccuracy})
     
    if epoch % 20 == 0 : ## Save the weights every 15 epochs 
        wandb.save("wts" + str(epoch) + ".npy")

wandb.save("wtsFinal.npy")
