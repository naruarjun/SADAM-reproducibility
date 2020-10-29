import torch
import torch.nn

class LogisticRegression(torch.nn.Module) :
    def __init__(self, inputSize, numClasses) :
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Sequential(
              torch.nn.Linear(inputSize, numClasses), 
              torch.nn.Softmax()
        )

    def forward(self, x):
        outputs = self.linear(x)
        return outputs 

class Layer4NN(torch.nn.Module) : 
    def __init__(self, inputSize, numClasses, channels = 3) : 
        super(Layer4NN, self).__init__()

        self.cnn_layers = torch.nn.Sequential(
            # Defining a 2D convolution layer , padding = 1 to retain dimensions 
            torch.nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(), 
            # Defining another 2D convolution layer : 32 filters again 
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout = torch.nn.Dropout(p = 0.25)
        self.fclayer = torch.nn.Sequential(
              torch.nn.Linear(inputSize ,128),  ## 128 hidden units 
              torch.nn.ReLU(),
              torch.nn.Dropout(p = 0.5),
              torch.nn.Linear(128 ,numClasses),  ## Final Linear Layer to input to the Softmax function
              torch.nn.Softmax() 
        )

    def forward(self, x) : 
        print(" Starting Model ")
        out = self.cnn_layers(x) 
        print(" CNN done ")
        out = self.dropout(out) 
        print(" Dropout done ", out.shape) 
        out = torch.flatten(out)
        return self.fclayer(out)
