"""
Created on Fri May 14 14:28:09 2021
@author: Gabor Asztalos

input: all patient csv with demog data and GEMuse signal output
output: custom NN model to predict rhythm classification for ECGs
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#dataset class for working with master dataset using muse features and demog
#   data to classify to labels in one category(rhythm,hyperytophy,morphology,longqt)
class MuseFeaturesDataset(Dataset): 
    def __init__(self, X, Y):
        X = X.clone().detach()
        self.x = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
#Neural network model with two hidden layers defined in layers
class MuseFeaturesModel(nn.Module):
    def __init__(self, input_size, layers, p=0.25):
        super().__init__()
        l1, l2, output_size = layers
        self.lin1 = nn.Linear(input_size, l1)
        self.lin2 = nn.Linear(l1, l2)
        self.lin3 = nn.Linear(l2, output_size)
        self.bn1 = nn.BatchNorm1d(l1)
        self.bn2 = nn.BatchNorm1d(l2)
        self.relu = nn.ReLU()
        self.drops = nn.Dropout(p)
        

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.drops(out)
        out = self.lin2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.drops(out)
        out = self.lin3(out)
        return out

#function to move data to device
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

#wrapper class for moving dataloader objects to device
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
#load dataset containing muse calculated ecg features, demog data, and labels
dataset = pd.read_csv(r'C:\Users\super\Documents\Learning\EKG\master.csv')

#set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#clean categroical data
categorical_columns = ['GENDER','RACE']
for cc in categorical_columns:
    dataset[cc].fillna('NA', inplace=True)
#clean numeric data
numeric_columns = ['Age','ATRIALRATE','P_RINTERVAL','QRSDURATION','Q_TINTERVAL',
                   'QTCCALCULATION','PAXIS','RAXIS','TAXIS','QRSCOUNT', 
                   'QONSET','QOFFSET','PONSET','POFFSET','TOFFSET']
for nc in numeric_columns:
    dataset[nc].fillna(0, inplace=True)
#set target column and find output size    
target_columns = ['rhythm_label']
dataset.loc[dataset['rhythm_label']==99, 'rhythm_label'] = 31 
freqs = dataset[target_columns].value_counts()
output_size = len(freqs)
#encode categorical data
for c in categorical_columns:
    dataset[c] = dataset[c].astype('category')
dataset['GENDER'] = dataset['GENDER'].cat.codes.values
dataset['RACE'] = dataset['RACE'].cat.codes.values

#turn features and labels into X,targets tensor
features = categorical_columns + numeric_columns
X = dataset[features]
X = torch.tensor(X.to_numpy(), dtype=torch.float32)
targets = torch.tensor(dataset[target_columns].values).flatten()

#test,train split of dataset
X_train, X_test, y_train, y_test = train_test_split(X, targets, 
                                    test_size=0.2,random_state=12)

#hyper parameters
batch_size = len(y_train)
layers = [128,64,output_size]
input_size = len(features)
learning_rate = 0.01
epochs = 200

#create custom datasets on train and test splits and put them in dataloader
#as well as on the device    
train_ds = MuseFeaturesDataset(X_train, y_train)
test_ds = MuseFeaturesDataset(X_test, y_test)
train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size,shuffle=False)
train_dl = DeviceDataLoader(train_dl, device)
test_dl = DeviceDataLoader(test_dl, device)

#create model and gradient functions
model = MuseFeaturesModel(input_size, layers)
to_device(model, device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#training loop
total_steps = len(train_dl)
aggregated_losses = []
for epoch in range(epochs):
    for i, (x,y) in enumerate(train_dl):
        #x, y = x.to(device), y.to(device)
        #forward pass
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print(f'epoch {epoch+1}/{epochs},step {i+1}/{total_steps}, loss = {loss.item():.4f}')
        aggregated_losses.append(loss.item())

#plot our loss function over epochs
plt.plot(range(epochs*total_steps), aggregated_losses)
plt.ylabel('Loss')
plt.xlabel('epoch');

#make final predictions with our trained model
preds = []
with torch.no_grad():
    for x,y in test_dl:
        outputs = model(x)
        prediction = np.argmax(outputs.to('cpu'), axis=1)
        #_, prediction = torch.max(outputs, 1)
        #prediction.cpu()
        preds.append(prediction.numpy())

flatten_list = [item for subl in preds for item in subl]
flatten_preds = np.array(flatten_list)
preds_tensor = torch.from_numpy(flatten_preds)
#print scores to evaluate model        
print(confusion_matrix(y_test,preds_tensor))
print(classification_report(y_test,preds_tensor))
print(accuracy_score(y_test, preds_tensor))

    

