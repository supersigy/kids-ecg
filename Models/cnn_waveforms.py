"""
Created on Wed Jun 23 10:44:17 2021
@author: Gabor Asztalos

input: folder of ECGs(csv) and ECG to target dictionary
output: CNN model to predict target on new ECGs
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

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

#custom dataloader class for ecg in dir to tensor and assigning target    
class WaveformsDataset(Dataset): 
    def __init__(self, path, targets):
        self.X = path
        self.y = targets
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        ecg = pd.read_csv(self.X[idx], header=None)
        target = self.y[idx]
        return torch.tensor(ecg.to_numpy(), dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

#CNN model 3x{conv->maxpool}-->3x{linear->relu}-->y_prediction    
class CNN(nn.Module):
    def __init__(self, leads, p=0.25):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(leads,22,kernel_size=11,padding=5)
        self.pool = nn.MaxPool1d(kernel_size=5,stride=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4,stride=4)
        self.conv2 = nn.Conv1d(22,44,kernel_size=11,stride=1)
        self.conv3 = nn.Conv1d(44,88,kernel_size=6,stride=4)
        self.lin1 = nn.Linear(88*6, 132)
        self.lin2 = nn.Linear(132, 33)
        self.lin3 = nn.Linear(33, 1)
        self.bn1 = nn.BatchNorm1d(132)
        self.bn2 = nn.BatchNorm1d(33)
        self.relu = nn.ReLU()
        self.drops = nn.Dropout(p)
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = self.conv3(out)
        out = self.pool2(out)
        out = out.view(-1,88*6)
        out = self.lin1(out)
        out = self.relu(out)
        out = self.bn1(out)
        out = self.drops(out)
        out = self.lin2(out)
        out = self.relu(out)
        out = self.bn2(out)
        out = self.drops(out)
        out = self.lin3(out)
        return out
    
#set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#read in dataset csv file, X=filepaths to waveform, y=QTCCALCULATION for ECG
df = pd.read_csv(r'C:\Users\super\Documents\Learning\EKG\file_to_target.csv')
df2 = df.iloc[:5000,:] #try code on 5k sample of ECGs
X = df2.ofile.values
y = df2.QTCCALCULATION.values

#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                    test_size=0.2,random_state=142)

#hyper parameters
batch_size = 20
learning_rate = 0.002
epochs = 20
leads = 11

#create custom datasets on train and test splits and put them in dataloader
#as well as on the device    
train_ds = WaveformsDataset(X_train, y_train)
test_ds = WaveformsDataset(X_test, y_test)
train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size,shuffle=False)
train_dl = DeviceDataLoader(train_dl, device)
test_dl = DeviceDataLoader(test_dl, device)

#create model and gradient functions
model = CNN(leads)
to_device(model, device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#training loop
total_steps = len(train_dl)
aggregated_losses = []
for epoch in range(epochs):
    for i, (x,y) in enumerate(train_dl):
        #forward pass(batch)
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        #backward pass(batch)
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

#create predictions on validation set
preds = []
with torch.no_grad():
    for x,y in test_dl:
        outputs = model(x).to('cpu')
        preds.append(outputs)

#put predictions and actual side by side        
flatten_list = [item.item() for subl in preds for item in subl]
y_hat = np.array(flatten_list)
compare = pd.DataFrame(np.column_stack((y_test,y_hat)),columns={'Y','Y_hat'})
compare['delta'] = abs(compare.Y-compare.Y_hat)

#plot prediction vs actual to surmise accuracy of model
plt.plot(y_test, y_hat)
plt.ylabel('Predicted')
plt.xlabel('Actual');