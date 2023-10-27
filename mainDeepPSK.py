import PSK

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

print("Libraries loaded")

# %% Model definition
G = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
   [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
   [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
   [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
#G = G[:, 0:8]

class SimpleNN(nn.Module):
    def __init__(self, input_size, sigma=1, hidden_size=64, 
                 keying="QPSK", encoding="polar",
                 encoding_matrix=G):
        super(SimpleNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigma = sigma
        self.encoding = encoding
        self.keying = keying
        self.encoding_matrix = encoding_matrix

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p=0.1)

        self.input_layer = nn.Linear(input_size, hidden_size)
        
        self.layer1 = nn.Linear(hidden_size, hidden_size//2)
        self.layer2 = nn.Linear(hidden_size//2, hidden_size//4)
        self.layer3 = nn.Linear(hidden_size//4, hidden_size//8)

        # self.batchnorm2 = nn.BatchNorm1d(hidden_size//3)

        self.output_layer = nn.Linear(hidden_size//8, input_size//2)
        self.sigmoid = nn.Sigmoid()  
            
    def forward(self, x):
        
        # The inputs are noised only when training
        if self.training:
            
            # All the inputs are encoded inside of the NN, to simplify the code
            if self.encoding == "polar":
                x = self.PolarCode(x=x)
                
            if self.keying == "BPSK":
                x = self.BPSK(x=x)
            elif self.keying == "QPSK":
                x = x
            
            x = self.GaussianNoise(x=x)

        # In evaluation mode, the inputs are considered already encoded and noised (= simulation)
        x = self.input_layer(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.relu(x)
        # x = self.batchnorm1(x)
        
        x = self.layer2(x)
        x = self.relu(x)
        # x = self.batchnorm2(x)
        
        x = self.layer3(x)
        x = self.relu(x)

        # x = self.dropout(x)
        
        x = self.output_layer(x)  
        x = self.relu(x)    
        x = self.sigmoid(x)

        return x - 0.01

    def BPSK(self, x):
        return x * 2 - 1
    
    def PolarCode(self, x):
        return torch.matmul(x, torch.tensor(self.encoding_matrix).float()) % 2
        
    def GaussianNoise(self, x):
        return x + torch.randn(x.size()) * self.sigma

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=16, output_size=8, hidden_size=128, 
                 sigma=1,
                 keying="BPSK", encoding="polar",
                 encoding_matrix=G):
        super(NeuralNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigma = sigma
        self.encoding = encoding
        self.keying = keying
        self.encoding_matrix = encoding_matrix

        self.decoder_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2), nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4), nn.ReLU(),
            nn.Linear(hidden_size//4, output_size), nn.Sigmoid()
        )

    def forward(self, x):

        # The inputs are noised only when training
        if self.training:
            # All the inputs are encoded inside of the NN, to simplify the code
            if self.encoding == "polar":
                x = self.PolarCode(x=x)
            # Modulation type
            if self.keying == "BPSK":
                x = self.BPSK(x=x)
            elif self.keying == "QPSK":
                x = x
            # Noise is added
            x = self.GaussianNoise(x=x)

        x = self.decoder_layers(x)

        return x

    def BPSK(self, x):
        return x * 2 - 1
    
    def PolarCode(self, x):
        return torch.remainder(torch.matmul(x, torch.tensor(self.encoding_matrix).float()), 2)
        
    def GaussianNoise(self, x):
        return x + torch.randn(x.size()) * self.sigma
    
class ConvNN(nn.Module):
    def __init__(self, input_size, sigma=1, hidden_size=64):
        super(ConvNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigma = sigma

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p=0.1)

        self.input_layer = nn.Linear(input_size, hidden_size)
        
        self.layer1 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size//2)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size//2)
        
        self.output_layer = nn.Linear(hidden_size//2, input_size)
        self.sigmoid = nn.Sigmoid()  
            
    def forward(self, x):
        
        if self.training:
            x = self.GaussianNoise(x=x)

        x = self.input_layer(x)
        x = self.relu(x)

        x = self.layer1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        
        x = self.output_layer(x)  
        x = self.relu(x)  

        x = self.sigmoid(x)

        return x

    def BPSK(self, x):
        return x * 2 - 1
    
    def PolarCode(self, x):
        return None
    
    def GaussianNoise(self, x):
        return x + torch.randn(x.size()) * self.sigma

class BERLoss(nn.Module):
    def __init__(self):
        super(BERLoss, self).__init__()

    def forward(self, outputs, targets):
        outputs = torch.round(outputs.clone().detach().requires_grad_(True))
        #targets = targets.view(-1)
        
        # bit_errors = torch.sum(torch.eq(outputs, targets)).item()
        intersection = (targets - outputs).sum() #(inputs * targets).sum()
        # print(intersection)
        # print(intersection.type())

        total_bits = targets.size()[0]
        ber = torch.div(intersection, total_bits)
        # print(ber)
        # print(ber.type())
        
        return ber

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        print(intersection)
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        print(dice)
        return 1 - dice
    
# %% Data preprocessing
if __name__ == "__main__" and "data" in sys.argv:

    word_length = 8
    m = 2**word_length
    combinations = np.arange(m, dtype=int)

    combinaison_array = np.zeros((2**word_length, word_length))
    for idx, number in tqdm(enumerate(combinations)):
        current_word = np.binary_repr(number, width=word_length)
        int_array = np.array([int(char) for char in current_word])
        combinaison_array[idx, :] = int_array

    print(combinaison_array.shape)

    train_features = torch.from_numpy(combinaison_array).float()
    train_labels = torch.from_numpy(combinaison_array).float()
    test_features = torch.from_numpy(combinaison_array).float()
    test_labels = torch.from_numpy(combinaison_array).float()

    # train_features = F.normalize(train_features)
    # test_features = F.normalize(test_features)

    print(train_features.size(), train_labels.size(), test_features.size(), test_labels.size())

    training_dataset = TensorDataset(train_features, train_labels)
    testing_dataset = TensorDataset(test_features, test_labels)

    train_dataloader = DataLoader(training_dataset, batch_size=m, shuffle=True)
    test_dataloader = DataLoader(testing_dataset, batch_size=m, shuffle=True)

# %% Model training
if __name__ == "__main__" and "training" in sys.argv:

    m, n = train_features.shape
    input_size = n*2
    output_size = n
    hidden_size = 128
    
    if ("nn" in sys.argv):
        model = NeuralNetwork(input_size=input_size, 
                              output_size=output_size,
                              hidden_size=hidden_size,
                              sigma=1, keying="BPSK", encoding="polar",
                              encoding_matrix=G)
        model_name = "NN"
        criterion = nn.MSELoss()
    else:
        # model = SimpleNN(input_size*2, hidden_size)
        # model_name = "SimpleNN"
        # criterion = BERLoss()
        pass

    if torch.cuda.is_available():
        print("CUDA is available.")
    else:
        print("CUDA is not available.")
        
    try:
        model.cuda()
        print("Model to gpu")
    except:
        print("Model to cpu")

    # criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = BERLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    model.train() 

    running_loss = 0.0
    losses_list = []
    num_epochs = int(input("Number of epochs : "))    
    for epoch in tqdm(range(num_epochs)):
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()  
            outputs = model(inputs)      
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            losses_list.append(loss.item())

               

    print(f"Final loss: {loss.item()}")

    plt.semilogy(losses_list)
    plt.xlabel("Epochs")
    plt.ylabel("Running loss")
    plt.show()

    torch.save(model.state_dict(), f"{model_name}_{num_epochs}.pth")
    print("Finished Training, model saved")

# %% Model testing
if __name__ == "__main__" and "testing" in sys.argv:

    if ("train" not in sys.argv) and ("simple" in sys.argv):
        num_epochs = int(input("Model's number of epochs to load : "))
        model = SimpleNN(input_size=word_length*2, hidden_size=64)
        model.load_state_dict(torch.load(f"SimpleNN_{num_epochs}.pth"))

    if ("train" not in sys.argv) and ("nn" in sys.argv):
        num_epochs = int(input("Model's number of epochs to load : "))
        model = NeuralNetwork()
        model.load_state_dict(torch.load(f"NN_{num_epochs}.pth"))

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for test_features, test_labels in test_dataloader:
            
            test_labels = test_labels.numpy()

            # Polar encoding
            test_features = torch.matmul(test_features, torch.tensor(G).float()) % 2
            # BPSK keying
            test_features = test_features * 2 - 1
            # Gaussian noise
            test_features = test_features + torch.randn(test_features.size()) * 1

            outputs = model(test_features)[:, 0:8]
            # print(outputs[0:10, :])
            predicted = torch.round(outputs.data).numpy()
            print(predicted[0:5, :])
            print(test_labels[0:5, :])

            # Success rate : 1 point per correct bit
            bit_wise_succes = np.sum(predicted == test_labels)

            # Success rate : 1 point per correct sequence
            word_wise_success = 0
            zeros = 0
            word_wise_success_idx = []
            for row_idx, row in tqdm(enumerate(test_labels)):
                if np.array_equal(row, predicted[row_idx, :]):
                    word_wise_success +=1
                    word_wise_success_idx.append(row_idx)
                if np.array_equal(np.zeros((1, 16)), predicted[row_idx, :]):
                    zeros +=1
            
            # print(predicted[0:5, :])
            # print(test_labels[0:5, :])
            print(bit_wise_succes, word_wise_success, zeros)

    df_predicted = pd.DataFrame(predicted)
    df_predicted.to_csv("output.csv")
    # print(np.unique(predicted.numpy(), return_counts=True))
    # print(np.unique(test_labels.numpy(), return_counts=True))
    # print(predicted.numpy()[0:10], test_labels.numpy().T[0, 0:10])
    
    # print(f"Accuracy on test set: {accuracy:.2f}%")

if __name__ == "__main__":
    bpsk_pipeline = PSK.bpsk_pipeline
    bpsk_class = PSK.BPSK(word_length=16, words_number=2000)
