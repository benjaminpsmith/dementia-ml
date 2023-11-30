import os
import random
import torch
from torchvision.transforms import v2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
import numpy as np
import os

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2

from functions import *
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import recall_score
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from augmentation import *

class MyDataset(Dataset):
    def __init__(self, X, Y, augment=False):
        self.X = X
        self.Y = Y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx]
        label = self.Y[idx]

        if self.augment == True:

            random_num = random.uniform(0.0, 1.0)

            if(random_num < 0.2):
                img = transforms['noise_gaussian'](img)
            elif(random_num < 0.4):
                img = transforms['noise_salt_pepper'](img)
            else:
                img = transforms['toTensor'](img)
        else:
            img = transforms['toTensor'](img)

        return img, label

def import_data():
    folder_paths = ['Dataset/Non_Demented/', 'Dataset/Very_Mild_Demented/', 'Dataset/Mild_Demented/', 'Dataset/Moderate_Demented/']
    classes = [r'Non demented', r'Very mildly demented', r'mild demented', r'moderate demented']

    X = []
    Y = []

    # Loop over the images to save them in the list
    for c, path in enumerate(folder_paths):
        items = os.listdir(path)
        for picture in items:
            file_path = os.path.join(path, picture)

            # Open the image and convert it to a NumPy array
            img = Image.open(file_path)
            array_representation = np.asarray(img)

            # Append the NumPy array to the list
            X.append(array_representation)
            Y.append(c)

    # Convert the list of image arrays to a NumPy array
    X = np.array(X)

    # Transpose to make each image a row
    X = X.reshape(X.shape[0], -1)

    # Normalize each row (i.e., each flattened image)
    X = MinMaxScaler().fit_transform(X)

    # Reshape back to the original shape
    X = X.reshape(len(X), 128, 128, 1)

    # Dynamically calculate the number of classes in the dataset
    num_classes = len(np.unique(Y))

    return X, Y, num_classes

def split_data(X, Y, train_per, val_per, test_per):

    assert train_per + val_per + test_per == 1.0, "Percentage split should sum to 1.0"

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(1 - train_per), random_state=42, stratify=Y)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=(test_per / (test_per + val_per)), random_state=42, stratify=Y_test)
 
    return X_train, X_val, X_test, Y_train, Y_val, Y_test