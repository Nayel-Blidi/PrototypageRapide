
import PSK

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

print("Libraries loaded")

# %% Model definition
def calculate_BER(predictions, original_data):
    errors = np.sum(predictions != original_data)
    total_bits = original_data.size
    ber = errors / total_bits
    return ber

G = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
   [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
   [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
   [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


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
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size//2), nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(hidden_size//2, output_size), nn.Sigmoid()
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
    
    
# %% Data preprocessing

word_length = 8
m = 2**word_length
combinations = np.arange(m, dtype=int)

combinaison_array = np.zeros((2**word_length, word_length))
for idx, number in tqdm(enumerate(combinations)):
    current_word = np.binary_repr(number, width=word_length)
    int_array = np.array([int(char) for char in current_word])
    combinaison_array[idx, :] = int_array



train_features = torch.from_numpy(combinaison_array).float()
train_labels = torch.from_numpy(combinaison_array).float()
test_features = torch.from_numpy(combinaison_array).float()
test_labels = torch.from_numpy(combinaison_array).float()


training_dataset = TensorDataset(train_features, train_labels)
testing_dataset = TensorDataset(test_features, test_labels)

train_dataloader = DataLoader(training_dataset, batch_size=m, shuffle=True)
test_dataloader = DataLoader(testing_dataset, batch_size=m, shuffle=True)

m, n = train_features.shape
input_size = n*2
output_size = n
hidden_size = 512

# %% recherche hyperparametre
from sklearn.model_selection import ParameterGrid
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, SGD, Adadelta, Adagrad

param_grid = {
    'hidden_size': [64, 128, 256, 512, 1024],
    'sigma': [0.1, 0.5, 1, 5, 10],
    'lr': [0.001, 0.005, 0.01, 0.05, 0.1],
    'criterion': [nn.MSELoss(), nn.L1Loss(), nn.BCELoss(), nn.CrossEntropyLoss()],
    'optimizer': [Adam, Adagrad],
}
nb_valeur=1
best_score = 0  # Initialisez avec une valeur élevée pour minimiser la perte
best_params = {}
matrice_resultats=[]
for params in ParameterGrid(param_grid):
    model = NeuralNetwork(input_size=input_size,
                          output_size=output_size,
                          hidden_size=params['hidden_size'],
                          sigma=params['sigma'],
                          keying="BPSK", encoding="polar",
                          encoding_matrix=G)
    
    criterion = params['criterion']
    optimizer = params['optimizer'](model.parameters(), lr=params['lr'])

    model.train()
    running_loss = 0.0
    num_epochs = 1000
    print(f"Ia numéro {nb_valeur} en cours")
    nb_valeur+=1
    print(f"utilisation des parametres : {params}")

    for epoch in range(num_epochs):
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Évaluez le modèle avec les paramètres actuels
    model.eval()
    correct_messages = 0
    with torch.no_grad():
        for test_features, test_labels in test_dataloader:
            test_labels = test_labels.numpy()
            test_features = torch.matmul(test_features, torch.tensor(G).float()) % 2
            test_features = test_features * 2 - 1
            test_features = test_features + torch.randn(test_features.size()) * 1
            outputs = model(test_features)[:, :8]
            predicted = torch.round(outputs.data).numpy()
            ber = calculate_BER(predicted, test_labels)
            count = 0
            for i in range(len(predicted)):
                if np.array_equal(predicted[i], test_labels[i]):
                    count += 1
            print(f"On a {count} message bien prédits sur {len(predicted)}")
            correct_messages += count
    
    matrice_resultats.append(params)
    matrice_resultats.append(f"resultat : {correct_messages}")
    matrice_resultats.append(f"ber : {ber*100:.2f}%")
    
    if correct_messages > best_score:
        best_score = correct_messages
        best_params = params
        best_score=count

print(f"Meilleurs hyperparamètres trouvés : {best_params} et son résultat est de : {best_score}")
