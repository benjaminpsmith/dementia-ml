import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from numpy import mean

def train_network(model, dataloader, criterion, optimizer, num_epochs, learning_rate_scheduler = None, dataloader_val = None, device = None):
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
    training_history = []
    validation_history = []
    loss_history_training = []
    loss_history_val = []

    for epoch in range(num_epochs):
        loss_epoch = [] # To keep track of the loss during one epoch
        for i, (x_train_minibatch, y_true) in enumerate(dataloader):
            x_train_minibatch = x_train_minibatch.float()
            x_train_minibatch = x_train_minibatch.to(device)
            y_true = y_true.to(device)

            # Forward pass
            y_pred = model(x_train_minibatch)
            loss = criterion(y_pred, y_true)
            loss_epoch.append(loss.item())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((i+1) % (n_total_steps // 2) == 0):
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

        # Save the history of the training process
        model.eval()
        training_history.append(get_accuracy_dataloader(model, dataloader))
        if dataloader_val:
            validation_history.append(get_accuracy_dataloader(model, dataloader_val))
        loss_history_training.append(mean(loss_epoch))
        model.train()

    if not validation_history:
        print("No validation dataloader was provided, therefore the validation history is empty. Validation data should be provided.")
    return loss_history_training, training_history, validation_history

def get_accuracy_dataloader(model, dataloader, ignore_warning=False):
    """
    Calculate the accuracy of a PyTorch model on a given DataLoader.

    Parameters:
    - model             : The PyTorch model to evaluate.
    - dataloader        : The DataLoader containing the evaluation data.
    - ignore_warning    : If True, ignores the warning about the model being in training mode.

    Returns The mean accuracy of the model on the provided DataLoader.
    """

    if (model.training & ~ignore_warning):
        import warnings
        warnings.warn("Model is in training mode, so dropout layers etc are considered. Set mode to mode.eval() ?")
    acc_list = []
    with torch.no_grad():  # Disable gradient computation during validation
        for i, (x_minibatch, y_true) in enumerate(dataloader):
            y_pred = model(x_minibatch)
            y_cls = torch.argmax(F.softmax(y_pred, dim=1), dim=1)
            acc_list.append(accuracy_score(y_true, y_cls))
    return mean(acc_list)