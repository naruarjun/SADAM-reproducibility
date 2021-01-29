import os
import torch
import wandb 
import numpy as np
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

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--batch_size', type = int, default = 64)

parser.add_argument('--load_weights', type = str2bool, default = False, help = 'Load previous weights or not')
parser.add_argument('--load_file', type = str, default = 'dne.pth')
parser.add_argument('--details_file', type = str, default = 'LossAccuracy.csv', help = 'Save values for plotting later')

parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--decay', type = float, default = 1e2, help = 'Whether Decay for LR is to be done')
parser.add_argument('--model', type = str, default = 'nn')
parser.add_argument('--loss', type = str, default = 'entropy')
parser.add_argument('--dataset', type = str, default = 'mnist')
parser.add_argument('--convex', type = str2bool, default = False, help = 'Whether loss fn is convex or not')
parser.add_argument('--optimizer', type = str, default = 'adam')

iterations = 1000
args = parser.parse_args()
wandb.init(project = args.dataset) 

config = wandb.config          # Initialize config
config.log_interval = 10     # how many batches to wait before logging training status
config.epochs = args.epochs             # number of epochs to train (default: 10)
config.batch_size =  args.batch_size         # input batch size for training (default: 64)
config.test_batch_size = args.batch_size   # input batch size for testing (default: 1000)

lossfn = PT.get_loss(args.loss)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
train_loader, test_loader, inpsize, classes, channels, instances = PT.get_dataset(args.dataset, args.batch_size) 

model = PT.get_model(args.model, inpsize, classes, channels)
optimizer = PT.get_optimizer(list(model.parameters()), args.optimizer, args.lr, args.convex, args.decay) 
model.to(device)

  
    
if args.model == "logistic":
  optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001, weight_decay=1e-5)
  
  if args.load_weights : 
    model = torch.load('model' + args.dataset +'.pt')
  else : 
    model = PT.train_model(model, lossfn, device, args.epochs, optimizer, train_loader) 
    torch.save(model, 'model' + args.dataset + '.pt')
 
  batch_size = instances // iterations 
  train_loader, _, inpsize, classes, channels, _ = PT.get_dataset(args.dataset, batch_size) 
  optimal_loss = PT.regret_calculation(train_loader, model, optimizer, lossfn, device, iterations, args.optimizer, args.convex, inpsize, classes, channels, args.decay)

else : 
    wandb.watch(model, log="all")
    _ = PT.train_model(model, lossfn, device, args.epochs, optimizer, train_loader, test_loader, False)
