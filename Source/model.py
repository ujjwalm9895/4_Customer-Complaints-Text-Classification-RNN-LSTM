import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Define a neural network model using an RNN layer
class RNNNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initialize the RNN model.

        :param input_size: Size of word embeddings
        :param hidden_size: Size of the hidden vector
        :param num_classes: Number of classes in the dataset
        """
        super(RNNNetwork, self).__init()
        # RNN Layer
        self.rnn = torch.nn.RNN(input_size=input_size,
                                hidden_size=hidden_size,
                                batch_first=True)
        # Linear Layer for classification
        self.linear = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_data):
        _, hidden = self.rnn(input_data)
        output = self.linear(hidden)
        return output

# Define a neural network model using an LSTM layer
class LSTMNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initialize the LSTM model.

        :param input_size: Size of word embeddings
        :param hidden_size: Size of the hidden vector
        :param num_classes: Number of classes in the dataset
        """
        super(LSTMNetwork, self).__init()
        # LSTM Layer
        self.rnn = torch.nn.LSTM(input_size=input_size,
                                 hidden_size=hidden_size,
                                 batch_first=True)
        # Linear Layer for classification
        self.linear = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_data):
        _, (hidden, _) = self.rnn(input_data)
        output = self.linear(hidden[-1])
        return output

def train(train_loader, valid_loader, model, criterion, optimizer, device,
          num_epochs, model_path):
    """
    Train the neural network model.

    :param train_loader: Data loader for the training dataset
    :param valid_loader: Data loader for the validation dataset
    :param model: The neural network model
    :param criterion: Loss function (e.g., CrossEntropyLoss)
    :param optimizer: The optimizer (e.g., Adam)
    :param device: CUDA or CPU
    :param num_epochs: Number of training epochs
    :param model_path: Path to save the trained model
    """
    best_loss = 1e8
    for i in range(num_epochs):
        print(f"Epoch {i+1} of {num_epochs}")
        valid_loss, train_loss = [], []
        model.train()
        # Training loop
        for batch_labels, batch_data in tqdm(train_loader):
            # Move data to GPU if available
            batch_labels = batch_labels.to(device)
            batch_labels = batch_labels.type(torch.LongTensor)
            batch_data = batch_data.to(device)
            # Forward pass
            batch_output = model(batch_data)
            batch_output = torch.squeeze(batch_output)
            # Calculate loss
            loss = criterion(batch_output, batch_labels)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Gradient update step
            optimizer.step()
        model.eval()
        # Validation loop
        for batch_labels, batch_data in tqdm(valid_loader):
            # Move data to GPU if available
            batch_labels = batch_labels.to(device)
            batch_labels = batch_labels.type(torch.LongTensor)
            batch_data = batch_data.to(device)
            # Forward pass
            batch_output = model(batch_data)
            batch_output = torch.squeeze(batch_output)
            # Calculate loss
            loss = criterion(batch_output, batch_labels)
            valid_loss.append(loss.item())
        t_loss = np.mean(train_loss)
        v_loss = np.mean(valid_loss)
        print(f"Train Loss: {t_loss}, Validation Loss: {v_loss}")
        if v_loss < best_loss:
            best_loss = v_loss
            # Save the model if validation loss improves
            torch.save(model.state_dict(), model_path)
        print(f"Best Validation Loss: {best_loss}")

def test(test_loader, model, criterion, device):
    """
    Test the trained neural network model.

    :param test_loader: Data loader for the test dataset
    :param model: The trained neural network model
    :param criterion: Loss function (e.g., CrossEntropyLoss)
    :param device: CUDA or CPU
    """
    model.eval()
    test_loss = []
    test_accu = []
    for batch_labels, batch_data in tqdm(test_loader):
        # Move data to the specified device (GPU or CPU)
        batch_labels = batch_labels.to(device)
        batch_labels = batch_labels.type(torch.LongTensor)
        batch_data = batch_data.to(device)
        # Forward pass
        batch_output = model(batch_data)
        batch_output = torch.squeeze(batch_output)
        # Calculate loss
        loss = criterion(batch_output, batch_labels)
        test_loss.append(loss.item())
        batch_preds = torch.argmax(batch_output, axis=1)
        # Move predictions to CPU if using GPU
        if torch.cuda.is_available():
            batch_labels = batch_labels.cpu()
            batch_preds = batch_preds.cpu()
        # Compute accuracy
        test_accu.append(accuracy_score(batch_labels.detach().numpy(),
                                        batch_preds.detach().numpy()))
    test_loss = np.mean(test_loss)
    test_accu = np.mean(test_accu)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accu}")
