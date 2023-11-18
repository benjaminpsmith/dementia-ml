import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
from numpy import mean

class EarlyStopper:
    def __init__(self, model, patience=5, min_delta=0):
        self.model = model
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_parameters = None

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_parameters = self.model.parameters()
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_network(model, dataloader, criterion, optimizer, num_epochs, learning_rate_scheduler = None, dataloader_val = None, device = None, early_stopper = None):
    '''
    Trains a given model with given data, criterion, optimizer on a specified number of epochs.
    Returns the loss, training (and validation history) for each epoch (at the end of each epoch) as a list.

    model                   : The PyTorch model to evaluate (torch.nn.Module).
    dataloader              : Should be an torch.utils.data.DataLoader instance, the data the network should be trained on
    num_epochs              : The number of epochs the network should be trained
    criterion               : The loss function should satisfy the inputs criterion(y_pred, y_true)
    optimizer               : The optimizer used to be used
    
    -------
    learning_rate_scheduler : If specified, learning_rate_scheduler is used
    dataloader_val          : If specified, dataloader is used for validation of the model
    device                  : If specified, specific device is used for optimization, usually no need to specify
    '''

    # Set device on where to optimize the network on
    if not device:    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device used for training: {device.type}')

    n_total_steps = dataloader.__len__()
    training_history = []           # Accuracy
    validation_history = []         # Accuracy
    loss_history_training = []      # Loss
    loss_history_val = []           # Loss

    model.train()
    for epoch in range(num_epochs):
        for i, (x_minibatch, y_true) in enumerate(dataloader):
            x_minibatch = x_minibatch.float()
            x_minibatch = x_minibatch.to(device)
            y_true = y_true.to(device)

            # Forward pass
            y_pred = model(x_minibatch)
            loss = criterion(y_pred, y_true)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i+1) % (n_total_steps // 2) == 0):
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

        # Update learning rate after 
        if learning_rate_scheduler:
            learning_rate_scheduler.step()

        # For evaluation of the model
        # Set the model in evaluation mode
        model.eval()

        # First calculate the models loss and accuracy AFTER the epoch (in evaluation mode)
        loss_history_training.append(get_loss_no_training(model, dataloader, criterion, device=device))
        training_history.append(get_accuracy_dataloader(model, dataloader))

        # In case of provided validation set calculate and save the loss and accuracy (in eval mode) for the validation set
        if dataloader_val:
            loss_history_val.append(get_loss_no_training(model, dataloader_val, criterion, device=device))
            validation_history.append(get_accuracy_dataloader(model, dataloader_val))

            # Check for early stopping, if provided
            if early_stopper is not None:
                if early_stopper.early_stop(loss_history_val[-1]):
                    model.parameters = early_stopper.best_parameters
                    print("Early stopping triggered and model parameters has been set to best validation parameters")
                    break  # You can break out of the training loop or take other actions

        # Set model in training mode again
        model.train()

    if not dataloader_val:
        print("No validation dataloader was provided, therefore the validation history is empty. Validation data should be provided.")
    return loss_history_training, loss_history_val, training_history, validation_history

def get_accuracy_dataloader(model, dataloader, ignore_warning=False):
    """
    Calculate the accuracy of a PyTorch model on a given DataLoader.

    Parameters:
    - model             : The PyTorch model to evaluate.
    - dataloader        : The DataLoader containing the evaluation data.
    - ignore_warning    : If True, ignores the warning about the model being in training mode.

    Returns The mean accuracy of the model on the provided DataLoader.
    """

    if (model.training & ignore_warning):
        import warnings
        warnings.warn("Model is in training mode, so dropout layers etc are considered. Set mode to mode.eval() ?")
    acc_list = []
    with torch.no_grad():  # Disable gradient computation during validation
        for i, (x_minibatch, y_true) in enumerate(dataloader):
            y_pred = model(x_minibatch)
            y_cls = torch.argmax(F.softmax(y_pred, dim=1), dim=1)
            acc_list.append(accuracy_score(y_true, y_cls))
    return mean(acc_list)

def get_loss_no_training(model, dataloader, criterion, device = None, ignore_warning = False):
    """
    Calculate the loss on a given dataset without updating the model's parameters.

    Parameters:
        model (torch.nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader instance for the dataset.
        criterion: The loss function that satisfies the inputs criterion(y_pred, y_true).
        device (torch.device, optional): The device to perform calculations on (default is 'cuda' if available, else 'cpu').
        ignore_warning (bool, optional): If True, ignore the warning about the model being in training mode.

    Returns:
        float: Mean loss over the dataset.
    """
    if not device:    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if (model.training & ignore_warning):
        import warnings
        warnings.warn("Model is in training mode, so dropout layers etc are considered. Set mode to mode.eval() ?")

    loss_epoch = [] # To keep track of the loss during one epoch
    with torch.no_grad():
        for i, (x_minibatch, y_true) in enumerate(dataloader):
            x_minibatch = x_minibatch.float()
            x_minibatch = x_minibatch.to(device)
            y_true = y_true.to(device)

            # Predict the label of the samples
            y_pred = model(x_minibatch)
            loss = criterion(y_pred, y_true)
            loss_epoch.append(loss.item())
    return mean(loss_epoch)

class custom_ConvNet(nn.Module):
    def __init__(
        self,
        input_channels,
        output_size, 
        n_conv_layers1, 
        n_filters1,
        kernel_size1,
        n_dense_layers,
        n_dense_initial_nodes,
        n_conv_layers2 = 0, 
        n_filters2 = 0,
        kernel_size2 = 0,
        operation_and_factor_filter_size = ('*', 1),
        operation_and_factor_dense_network = ('*', 1),
        dropout_rate = 0,
        pooling = None,
        activation_function = F.leaky_relu,
        ):

        super(custom_ConvNet, self).__init__()

        # Dynamically get the operation and the factor with which the network will later shrink/grow layer to layer
        valid_operations = {'+': lambda x, y: x + y, '-': lambda x, y: x - y, '*': lambda x, y: x * y, '/': lambda x, y: x // y,'//': lambda x, y: x // y}
        operation_conv, factor_conv =  valid_operations.get(operation_and_factor_filter_size[0], None), operation_and_factor_filter_size[1]
        operation_dnn, factor_dnn =  valid_operations.get(operation_and_factor_dense_network[0], None), operation_and_factor_dense_network[1]

        # Save activation function for forward call
        self.activation_function = activation_function
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2

        # Initialize modules and dropout etc
        self.dnn = nn.ModuleList()
        self.cnn1 = nn.ModuleList()
        self.cnn2 = nn.ModuleList() if n_conv_layers2 > 0 else None
        self.dropout2d = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.pool = nn.MaxPool2d(pooling[0], pooling[1]) if pooling is not None else None

        # Initialize dummy parameters for first forward pass
        self.input_size = None
        self.input_dnn = None
        self.n_dense_initial_nodes = n_dense_initial_nodes

        # Initialize the dense network
        for i in np.arange(n_dense_layers): 
            if i == 0:
                # The input size of the first linear layer will be calculated dynamically in the first forward call
                self.dnn.append(nn.Linear(1, n_dense_initial_nodes))
                nodes = n_dense_initial_nodes

            if ((i > 0) & (i < n_dense_layers - 1)):
                output = operation_dnn(nodes, factor_dnn)
                self.dnn.append(nn.Linear(nodes, operation_dnn(nodes, factor_dnn)))
                nodes = output

            if (i == n_dense_layers - 1):
                print(f'Size last layer before output: {nodes}')
                self.dnn.append(nn.Linear(nodes, output_size))

        # Build the first convolutional network part
        for i in np.arange(n_conv_layers1): 
            if i == 0:
                self.cnn1.append(nn.Conv2d(input_channels, n_filters1, kernel_size1))
                previous_filters = n_filters1

            elif ((i > 0) & (i < n_conv_layers1)):
                output_filters = operation_conv(nodes, factor_conv)
                self.cnn1.append(nn.Conv2d(previous_filters, output_filters, kernel_size1))
                nodes = output_filters

        # Build the second convolutional network part
        for i in np.arange(n_conv_layers2): 
            if i == 0:
                self.cnn2.append(nn.Conv2d(input_channels, n_filters2, kernel_size2))
                previous_filters = n_filters2

            elif ((i > 0) & (i < n_conv_layers2)):
                output_filters = operation_conv(nodes, factor_conv)
                self.cnn2.append(nn.Conv2d(previous_filters, output_filters, kernel_size2))
                nodes = output_filters

    def forward(self, x):
        x1 = x
        for layer in self.cnn1[:-1]:
            if ((self.pool is not None) & (self.dropout2d is not None)):
                x1 = self.pool(self.activation_function(self.dropout2d(layer(x1))))
            elif (self.pool is not None) & (self.dropout2d is None):
                x1 = self.pool(self.activation_function(layer(x1)))
            elif (self.pool is None) & (self.dropout2d is not None):
                x1 = self.activation_function(self.dropout2d(layer(x1)))
            else:
                x1 = self.activation_function(layer(x1))
        x1 = x1.view(-1, x1.shape[1] * x1.shape[2] * x1.shape[3])
        # x1 = x1.view(-1, x1.numel() // x1.size(0))

        if self.cnn2 is not None:
            x2 = x
            for layer in self.cnn2[:-1]:
                if ((self.pool is not None) & (self.dropout2d is not None)):
                    x2 = self.pool(self.activation_function(self.dropout2d(layer(x2))))
                elif (self.pool is not None) & (self.dropout2d is None):
                    x2 = self.pool(self.activation_function(layer(x2)))
                elif (self.pool is None) & (self.dropout2d is not None):
                    x2 = self.activation_function(self.dropout2d(layer(x2)))
                else:
                    x2 = self.activation_function(layer(x2))
            x2 = x2.view(-1, x2.shape[1] * x2.shape[2] * x2.shape[3])
            # x2 = x2.view(-1, x2.numel() // x2.size(0))
        else:
            x2 = None

        # Put convolutional networks together
        x = torch.cat((x1, x2), dim=1) if x2 is not None else x1
        # Set input dimension of first dense layer
        self.dnn[0] = nn.Linear(x.shape[1], self.n_dense_initial_nodes)
        
        for layer in self.dnn[:-1]:
            if self.dropout is not None:
                x = self.dropout(self.activation_function(layer(x)))
            else:
                x = self.activation_function(layer(x))

        # Output layer
        x = self.dnn[-1](x)
        return x
    
class Custom_DNN(nn.Module):
    def __init__(self, input_size, initial_nodes, output_size, n_layers = 5, operation_and_factor = ('/', 2), dropout_rate=0):
        super(Custom_DNN, self).__init__()
        
        # Dynamically get the operation and the factor with which the network will later shrink/grow layer to layer
        # For simple dense neural networks an increase in nodes is very uncommon though
        valid_operations = {'+': lambda x, y: x + y, '-': lambda x, y: x - y, '*': lambda x, y: x * y, '/': lambda x, y: x // y,'//': lambda x, y: x // y}
        operation, factor =  valid_operations.get(operation_and_factor[0], None), operation_and_factor[1]

        self.linears = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)

        for i in np.arange(n_layers): 
            if i == 0:
                self.linears.append(nn.Linear(input_size, initial_nodes))
                nodes = initial_nodes

            if ((i > 0) & (i < n_layers - 1)):
                output = operation(nodes, factor)
                self.linears.append(nn.Linear(nodes, operation(nodes, factor)))
                nodes = output

            if (i == n_layers - 1):
                print(f'Size last layer before output: {nodes}')
                self.linears.append(nn.Linear(nodes, output_size))

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = self.dropout(F.leaky_relu(layer(x)))

        # Output layer (no activation for regression tasks, modify as needed)
        x = self.linears[-1](x)
        return x
