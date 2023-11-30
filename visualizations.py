import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset

from functions import *

train_size = 0.7
test_size = 1 - train_size


# Create an empty array to store the image arrays and class
X = []
Y = []

# Define the folder paths containing the images
folder_paths = ['Dataset/Non_Demented/', 'Dataset/Very_Mild_Demented/', 'Dataset/Mild_Demented/', 'Dataset/Moderate_Demented/']
classes = [r'Non demented', r'Very mildly demented', r'mild demented', r'moderate demented']

# Loop over the images to save them in the list
for path in folder_paths:
    c = folder_paths.index(path)
    items = os.listdir(path)
    for picture in items:
        file_path = os.path.join(path, picture)
        # Open the image and convert it to a NumPy array
        img = Image.open(file_path)
        array_representation = np.asarray(img)

        # Append the NumPy array to the list
        X.append(array_representation)
        Y.append(c)

# Convert the list of image arrays to a NumPy arrayF
X = np.array(X)

# Transpose to make each image a row

X = X.reshape(X.shape[0], -1)

# Normalize each row (i.e., each flattened image)
X = MinMaxScaler().fit_transform(X)

# Reshape back to the original shape
X = X.reshape(len(X), 128, 128, 1)

# Dynamically calculate the number of classes in dataset
num_classes = len(np.unique(Y))

# Assuming you have a class named MyDataset for your dataset
class MyDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.Y[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

# Define transformations, you can adjust these based on your needs
transform = v2.Compose([
    v2.ToImage(), 
    v2.ToDtype(torch.float32, scale=True)
])

# Create an instance of your dataset
dataset = MyDataset(X=X, Y=Y, transform=transform)

train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [int(train_size * len(dataset)), len(dataset) - int(train_size * len(dataset))]
    )

train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [int(train_size * len(train_dataset)), len(train_dataset) - int(train_size * len(train_dataset))]
    )

# Define DataLoader for training and test sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# To load model
criterion = nn.CrossEntropyLoss()
def criterion_function(y_pred, y_cls, y_true):
    return criterion(torch.tensor(y_pred), torch.tensor(y_true))
def accuracy_function(y_pred, y_cls, y_true):
    return accuracy_score(y_true, y_cls)
def recall_function(y_pred, y_cls, y_true):
    return recall_score(y_true, y_cls, average='macro')

# Load model & make prediction

test_model = torch.load('test_model.pth')

y_pred, y_true = test_model.evaluate(test_loader)
y_cls = np.argmax(y_pred, axis=1)

#Note, this code is taken straight from the SKLEARN website, a nice way of viewing confusion matrix.
import itertools
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    thresh = 1
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

confusion_mtx = confusion_matrix(y_true, y_cls)
# plot the confusion matrix

labels = ['Non \n demented', 'Very mildly \n demented', 'Mildly \n demented', 'Moderately \n demented']

plt.figure(figsize=(8,8))
plot_confusion_matrix(confusion_mtx, classes = labels)
plt.figure(figsize=(8,8))
plot_confusion_matrix(confusion_mtx, classes = labels, normalize=False)

print(classification_report(y_true, y_cls))

