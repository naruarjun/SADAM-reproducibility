import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type = int, default = 200)
parser.add_argument('--batch_size', type = int, default = 64)

parser.add_argument('--load_file', type = str, default = 'dne.pth')
parser.add_argument('--details_file', type = str, default = 'LossAccuracy.csv', help = 'Save values for plotting later')
parser.add_argument('--log_dir', type = str, default = 'logs')

parser.add_argument('--model', type = str, default = 'logistic')
parser.add_argument('--loss', type = str, default = 'entropy')
parser.add_argument('--dataset', type = str, default = 'mnist')
parser.add_argument('--optimizer', type = str, default = 'adam')

args = parser.parse_args()

print(args) 
print(args.epochs) 
