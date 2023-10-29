import QAM

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

def PipelineSixteenQAM(words_number=10000, noise=True, sigma=1):
    """
    Function \n
    Returns a randomly generated bits sequence (Y) and simulated sampled signal (X).
    """
    qam_class = QAM.SixteenQAM_light(words_number=words_number, sigma=sigma, visualizations=False)
    Fs = qam_class.Fs
    Fc = qam_class.Fc
    T = qam_class.timeResolution

    Y = qam_class.sequenceGenerator()
    # qam_class.QAM()
    X = qam_class.signalModulation()
    if noise:
        X = qam_class.gaussianNoise()
    # X = qam_class.signalSampling()[:, ::round(T/(Fs/Fc))]
    return X, Y

class LayeredNN(nn.Module):
    def __init__(self, input_size, output_size=4, sigma=1, hidden_size=64):
        super(LayeredNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sigma = sigma

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p=0.1)

        self.input_layer = nn.Linear(input_size, hidden_size)
        
        self.layer1 = nn.Linear(hidden_size, hidden_size//2)
        self.batchnorm1 = nn.BatchNorm1d(num_features=hidden_size//2)
        self.layer2 = nn.Linear(hidden_size//2, hidden_size//4)
        self.batchnorm2 = nn.BatchNorm1d(num_features=hidden_size//4)
        self.layer3 = nn.Linear(hidden_size//4, hidden_size//8)
        self.batchnorm3 = nn.BatchNorm1d(num_features=hidden_size//8)


        self.output_layer = nn.Linear(hidden_size//8, output_size)
        self.sigmoid = nn.Sigmoid()  
            
    def forward(self, x):
        
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.relu(x)
        x = self.batchnorm3(x)
        x = self.dropout(x)
        
        x = self.output_layer(x)  
        x = self.relu(x)   
        # x = self.dropout(x)

        x = self.sigmoid(x)

        return x

class SequentialLayeredNN(nn.Module):
    def __init__(self, input_size, output_size=4, sigma=1, hidden_size=64):
        super(SequentialLayeredNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sigma = sigma

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p=0.1)

        self.decoder_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2), nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4), nn.ReLU(),
            nn.Linear(hidden_size//4, output_size), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder_layers(x)  

        return x

class ConvNN(nn.Module):
    def __init__(self, input_size, output_size=4, sigma=1, hidden_size=64, num_filters=16):
        super(ConvNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_filters = num_filters
        self.sigma = sigma

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=None, padding=1)
        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout1d(p=0.1)

        self.input = nn.Linear(in_features=input_size, out_features=hidden_size)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=3)

        # self.output = nn.Linear(in_features=num_filters*8-8, out_features=output_size)
        self.output = nn.Linear(in_features=num_filters*2, out_features=output_size)

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = torch.max(x, dim=2)[0]

        # x = self.dropout(x)

        x = self.output(x)
        x = self.sigmoid(x)

        return x

class RNN(nn.Module):
    def __init__(self, input_size, output_size=4, sigma=1, hidden_size=64):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sigma = sigma


    def forward(self):
        pass

# %% Model training
if __name__ == "__main__" and "training" in sys.argv:

    # Dummy call to retreive datasets dimensions
    X_train, Y_train = PipelineSixteenQAM(1)
    m, n = X_train.shape
    input_size = n
    output_size = n
    hidden_size = 128
    
    if "ConvNN" in sys.argv:
        model = ConvNN(input_size=input_size, 
                        hidden_size=hidden_size)
        model_name = "SixteenQAM_ConvNN"
    elif "LayeredNN" in sys.argv:
        model = LayeredNN(input_size=input_size, 
                            hidden_size=hidden_size)
        model_name = "SixteenQAM_LayeredNN"
    elif "SequentialNN" in sys.argv:
        hidden_size=512
        model = SequentialLayeredNN(input_size=input_size, 
                                    hidden_size=hidden_size)
        model_name = "SixteenQAM_SequentialLayeredNN"
    else:
        model = LayeredNN(input_size=input_size, 
                            hidden_size=hidden_size)
        model_name = "SixteenQAM_LayeredNN"

    if torch.cuda.is_available():
        print("CUDA is available.")
        model.cuda()
    else:
        print("CUDA is not available.")

    # criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01,) #momentum=0.5)
    model.train() 

    running_loss = 0.0
    losses_list = []
    num_epochs = int(input("Number of epochs : "))    
    for epoch in tqdm(range(num_epochs)):

        words_number = 1000
        X, Y = PipelineSixteenQAM(words_number=words_number, sigma=1)
        train_dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
        train_dataloader = DataLoader(train_dataset, batch_size=words_number, shuffle=True)

        # print(X.shape)
        # print(Y.shape)
        # plt.plot(X[0,:])
        # plt.title(Y[0,:])
        # plt.show()

        for inputs, targets in train_dataloader:
            if "ConvNN" in sys.argv:
                inputs = torch.unsqueeze(input=inputs, dim=1)
            optimizer.zero_grad()  
            outputs = model(inputs)  

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            losses_list.append(loss.item())

    print(inputs[0]) 
    print(targets[0:5])
    print(outputs[0:5])   
    print(f"Final loss: {loss.item()}")

    # plt.plot(losses_list)
    # plt.xlabel("Epochs")
    # plt.ylabel("Running loss")
    # plt.show()

    torch.save(model.state_dict(), f"{model_name}_{num_epochs}.pth")
    print("Finished Training, model saved")

# %% Model testing
if __name__ == "__main__" and "testing" in sys.argv:

    words_number = 10000
    X, Y = PipelineSixteenQAM(words_number=words_number, sigma=1)
    test_dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    test_dataloader = DataLoader(test_dataset, batch_size=words_number, shuffle=True)

    # Dummy call to retreive datasets dimensions
    X_train, Y_train = PipelineSixteenQAM(1)
    m, n = X_train.shape
    input_size = n
    output_size = n
    hidden_size = 128

    if ("training" not in sys.argv) and ("LayeredNN" in sys.argv):
        num_epochs = int(input("Model's number of epochs to load : "))
        model = LayeredNN(input_size=input_size, hidden_size=128)
        model.load_state_dict(torch.load(f"SixteenQAM_LayeredNN_{num_epochs}.pth"))

    if ("training" not in sys.argv) and ("ConvNN" in sys.argv):
        num_epochs = int(input("Model's number of epochs to load : "))
        model = ConvNN(input_size=input_size, hidden_size=128)
        model.load_state_dict(torch.load(f"SixteenQAM_ConvNN_{num_epochs}.pth"))

    if ("training" not in sys.argv) and ("SequentialNN" in sys.argv):
        num_epochs = int(input("Model's number of epochs to load : "))
        model = SequentialLayeredNN(input_size=input_size, hidden_size=128)
        model.load_state_dict(torch.load(f"SixteenQAM_SequentialLayeredNN_{num_epochs}.pth"))

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for test_features, test_labels in test_dataloader:
            
            test_labels = test_labels.numpy()
            if "ConvNN" in sys.argv:
                test_features = torch.unsqueeze(input=test_features, dim=1)

            outputs = model(test_features)
            predicted = torch.round(outputs.data).numpy()

            # Success rate : 1 point per correct bit
            bit_wise_succes = np.sum(predicted == test_labels)

            # Success rate : 1 point per correct sequence
            word_wise_success = 0
            zeros = 0
            word_wise_success_idx = []
            for row_idx, row in enumerate(test_labels):
                if np.array_equal(row, predicted[row_idx, :]):
                    word_wise_success +=1
                    word_wise_success_idx.append(row_idx)
                if np.array_equal(np.zeros((1, 16)), predicted[row_idx, :]):
                    zeros +=1

            bit_wise_accuracy = round(bit_wise_succes/np.size(test_labels), 2)
            word_wise_accuracy = round(word_wise_success/test_labels.shape[0], 2)
            print(predicted[0:5, :])
            print(test_labels[0:5, :])
            print(f"correct bits = {bit_wise_succes}, correct words = {word_wise_success}")
            print(f"bit success rate = {bit_wise_accuracy}, word success rate = {word_wise_accuracy}")

    df_predicted = pd.DataFrame(predicted)
    df_predicted.to_csv("output.csv")
