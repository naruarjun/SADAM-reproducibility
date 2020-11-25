import os
import torch
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
parser.add_argument('--epochs', type = int, default = 200)
parser.add_argument('--batch_size', type = int, default = 64)

parser.add_argument('--load_weights', type = str2bool, default = False, help = 'Load previous weights or not')
parser.add_argument('--load_file', type = str, default = 'dne.pth')
parser.add_argument('--details_file', type = str, default = 'LossAccuracy.csv', help = 'Save values for plotting later')
parser.add_argument('--log_dir', type = str, default = 'logs')

parser.add_argument('--model', type = str, default = 'nn')
parser.add_argument('--loss', type = str, default = 'entropy')
parser.add_argument('--dataset', type = str, default = 'mnist')
parser.add_argument('--convex', type = str2bool, default = False, help = 'Whether loss fn is convex or not')
parser.add_argument('--optimizer', type = str, default = 'adam')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
safe_device = torch.device("cpu") 
lossfn = PT.get_loss(args.loss)
train_loader, test_loader, input_size, num_classes, channels = PT.get_dataset(args.dataset, args.batch_size) 

model = PT.get_model(args.model, input_size, num_classes, channels)
optimizer = PT.get_optimizer(list(model.parameters()), args.optimizer, args.convex) 
model.to(device) 

if args.load_weights : 
    model.load_state_dict(torch.load(args.load_file))

### Training Code to be written 
epoch_loss, epoch_accuracy = [], []
for epoch in tqdm(range(int(args.epochs))) :
    best_accuracy, curr_loss = 0, 0 
    
    for iteration, data in tqdm(enumerate(train_loader)) :
        images, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = lossfn(outputs, labels)
        loss.backward()
        optimizer.step()

        if iteration % 500 == 0 : # Calculate Accuracy
            correct, total = 0, 0
            for imagesT, labelsT in test_loader :
                imagesT, labelsT = imagesT.to(device), labelsT.to(device)
                outputs = model(imagesT)
                _, predicted = torch.max(outputs.data, 1)
                total = total + labels.size(0)
                # for gpu, bring the predicted and labels back to cpu for python operations to work
                labelsT = labelsT.to(safe_device)  
                correct = correct + (predicted == labelsT).sum()
            accuracy = 100 * correct/total
            print("Iteration {}. Loss: {}. Accuracy: {}.".format(iteration, loss.item(), accuracy))
            if accuracy > best_accuracy :
                best_accuracy, curr_loss = accuracy, loss.item()

    epoch_loss.append(curr_loss)
    epoch_accuracy.append(best_accuracy)

    if epoch % 15 == 0 : ## Save the weights every 15 epochs 
        torch.save(model.state_dict(), os.path.join(args.log_dir, "wts" + str(epoch) + ".pth"))
        print("Weights saved at epoch number", epoch) 

PT.write_details(epoch_loss, epoch_accuracy, args.details_file)
print(" Exiting now ") 
