import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
This file uses the CIFAR10 dataset and simple CNN to classify the images in the dataset. There are 10 image classes
and the data is divided into train/validate datasets and tested on the test dataset
"""

# Define the transforms used to increase model performance
train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225]), 
                                    transforms.RandomVerticalFlip()])

test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

train_set = datasets.CIFAR10(root = './data', train=True, download=True, transform=train_transforms)
test_set = datasets.CIFAR10(root = '/.data', train=False, download=True, transform=test_transforms)

train_size = int(0.8 * len(train_set))  # 80% training
val_size = len(train_set) - train_size  # 20% validation

# Split dataset
train_set, valid_set = torch.utils.data.random_split(train_set, [train_size, val_size])


# Create a dataloader for the train and test data
# ----- set True or False for shuffing ---
trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True )
testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False )
validloader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=False )

num_classes = len(test_set.classes)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layer=nn.Sequential(
            nn.Conv2d(3, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.25),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.25))

        self.fc_layer = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.cnn_layer(x)
        x = x.flatten(1)
        out = self.fc_layer(x)
        return out


class TrainModel():
    def __init__(self, model, criterion,  optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
    
    def train_epoch(self, trainloader):
        # Set the model to training mode
        model = self.model.train()
        correct=0
        running_loss = 0.0
        progress_bar=tqdm(trainloader)
        for data in progress_bar:
            batch, labels = data  # separate inputs and labels (outputs)
            batch, labels = batch.to(self.device), labels.to(self.device)  # puts the data on the GPU

            self.optimizer.zero_grad() # clear the gradients in model parameters
            outputs = model(batch) # put data into model to predict
            loss = self.criterion(outputs, labels) # calculate loss between prediction and true labels
            loss.backward() # back propagation: pass the loss
            self.optimizer.step()  # iterate over all parameters in the model with requires_grad=True and update their weights.

            # compute training statistics
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() # sum total loss in current epoch for print later
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_set)
        avg_acc = correct / len(train_set)

        return avg_loss, avg_acc

    def eval_model(self, validloader):
        model=self.model.eval() # puts the model in validation mode
        with torch.no_grad():
            loss_val = 0.0
            correct_val = 0
        for data in tqdm(validloader):
            batch, labels = data
            batch, labels = batch.to(self.device), labels.to(self.device)
            outputs = model(batch)
            loss = self.criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            loss_val += loss.item()
        avg_loss_val = loss_val / len(valid_set)
        avg_acc_val = correct_val /len(valid_set)

        return avg_loss_val, avg_acc_val

    def test(self, testloader):
        correct = 0
        model=self.model.eval()
        with torch.no_grad(): # no gradient calculation
            for data in testloader:
                batch, labels = data
                batch, labels = batch.to(self.device), labels.to(self.device)
                outputs = model(batch)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        return ('Accuracy on the test images: %.2f %%' % (100 * correct / len(test_set)))
    
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Determine whether a GPU is available
criterion = nn.CrossEntropyLoss().to(device)  # We use Cross Entropy Loss, as this is a classification task

cnn = CNN()
cnn.to(device) # send model to GPU
optimizer = optim.Adam(cnn.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

model = TrainModel(cnn, criterion, optimizer, device)
print("Loaded in training model and commecning training: ")

epoch=10
#acc log for graph
simp_acc_hist = []
simp_acc_hist_val =[]
for e in range(epoch):
  print(f'Epoch {e + 1}/{epoch}')
  print('-' * 10)
  simple_train_loss ,simple_train_acc= model.train_epoch(trainloader)
  simp_acc_hist.append(simple_train_acc)
  print(f'Train loss {simple_train_loss} accuracy {simple_train_acc}')

  simple_val_loss, simple_val_acc = model.eval_model(validloader)
  simp_acc_hist_val.append(simple_val_acc)

  print(f'Val loss {simple_val_loss} accuracy {simple_val_acc}')
  print()