import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

class RNN(nn.Module):
    def __init__(self, input_size=300, hidden_size=128, num_layers=1, output_size=2, fc_layers_sizes=[]):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Define fully connected layers
        fc_layers = []
        prev_size = hidden_size
        for size in fc_layers_sizes:
            fc_layers.append(nn.Linear(prev_size, size))
            fc_layers.append(nn.ReLU())
            prev_size = size
        fc_layers.append(nn.Linear(prev_size, output_size))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)

        # Pass through fully connected layers
        out = self.fc(out[:, -1, :])
        return out

class RNNWithPooling(nn.Module):
    def __init__(self, input_size=300, hidden_size=128, num_layers=1, output_size=2, fc_layers_sizes=[], pooling_type='avg'):
        super(RNNWithPooling, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.pooling_type = pooling_type

        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Define fully connected layers
        fc_layers = []
        prev_size = hidden_size
        for size in fc_layers_sizes:
            fc_layers.append(nn.Linear(prev_size, size))
            fc_layers.append(nn.ReLU())
            prev_size = size
        fc_layers.append(nn.Linear(prev_size, output_size))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)

        # Apply pooling
        if self.pooling_type == 'avg':
            sentence_representation = torch.mean(out, dim=1)  # Average across the time dimension
        elif self.pooling_type == 'max':
            sentence_representation, _ = torch.max(out, dim=1)  # Max across the time dimension
        else:
            raise ValueError("Invalid pooling type. Choose 'avg' or 'max'.")

        # Pass through fully connected layers
        out = self.fc(sentence_representation)
        return out

class RNNWithConcatPooling(nn.Module):
    def __init__(self, input_size=300, hidden_size=128, num_layers=1, output_size=2, fc_layers_sizes=[]):
        super(RNNWithConcatPooling, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Define the RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # Define fully connected layers
        fc_layers = []
        prev_size = hidden_size * 2  # Concatenation of last hidden state and max pooling
        for size in fc_layers_sizes:
            fc_layers.append(nn.Linear(prev_size, size))
            fc_layers.append(nn.ReLU())
            prev_size = size
        fc_layers.append(nn.Linear(prev_size, output_size))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)

        # Extract last hidden state and apply max pooling
        last_hidden = out[:, -1, :]
        max_pooling, _ = torch.max(out, dim=1)

        # Concatenate last hidden state with max-pooled hidden states
        sentence_representation = torch.cat((last_hidden, max_pooling), dim=1)

        # Pass through fully connected layers
        out = self.fc(sentence_representation)
        return out

def train_model_rnn(model, train_loader, val_loader, num_epochs, device, model_name, learning_rate=0.005, optimizer_type='Adam'):
    """
    Train the model with a given training and validation loader.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device): Device to use for training (CPU or GPU).
        model_name (str): The name of the model (used for saving checkpoints).
        learning_rate (float): Learning rate for the optimizer.
        optimizer_type (str): Type of optimizer to use ('Adam', 'SGD', etc.).

    Returns:
        history (dict): Dictionary containing training and validation loss history.
        early_stop_epoch (int): The epoch at which early stopping was triggered (None if not triggered).
        early_stop_history (dict): Dictionary containing the training and validation loss history at the early stopping point.
    """

    print("Learning Rate: ", learning_rate)
    # Move model to the appropriate device
    model.to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) if model.fc[-1].out_features > 1 else nn.MSELoss()

    # Define the optimizer based on optimizer_type
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.01)
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:  # Default to AdamW
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    # Initialize EMA model
    ema = torch.optim.swa_utils.AveragedModel(model)

    # Initialize history dictionary
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    early_stop_history = None

    # Early stopping parameters
    best_val_acc = 0
    patience = 7
    patience_counter = 0
    early_stop_epoch = None

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Training Phase
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # Move inputs and labels to device
            inputs, labels = inputs.to(device), labels.squeeze().to(device)

            # Apply mixup augmentation after 3 epochs
            if epoch > 3:
                alpha = 0.2
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(inputs.size(0)).to(device)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                inputs = mixed_inputs

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss with mixup if applicable
            if epoch > 3:
                loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])
            else:
                loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update EMA model
            ema.update_parameters(model)

            # Track training loss and accuracy
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total

        # Validation Phase with EMA model
        ema.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move inputs and labels to device
                inputs, labels = inputs.to(device), labels.squeeze().to(device)

                # Forward pass
                outputs = ema(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Update history
        history['train_loss'].append(total_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # Print epoch stats
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Training Loss: {total_loss / len(train_loader):.4f}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Accuracy: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')

        # Early stopping and model checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f'New best validation accuracy! Saving model...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'history': history
            }, f'best_{model_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch + 1}')
                early_stop_epoch = epoch + 1
                early_stop_history = {
                    'train_loss': history['train_loss'][:early_stop_epoch],
                    'train_acc': history['train_acc'][:early_stop_epoch],
                    'val_acc': history['val_acc'][:early_stop_epoch],
                    'lr': history['lr'][:early_stop_epoch]
                }
                break

    return history, early_stop_epoch, early_stop_history

def train_model_multiple_optimizers(model_class, train_loader, val_loader, device, model_name, num_runs=3, num_epochs=30):
    """
    Train the model multiple times with different optimizers and plot the training and validation accuracies.

    Args:
        model_class (type): The class of the model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to use for training (CPU or GPU).
        model_name (str): The name of the model (used for saving checkpoints).
        num_runs (int): Number of times to train the model per optimizer.
        num_epochs (int): Number of epochs to train the model in each run.
    """
    optimizers = ['SGD', 'RMSprop', 'Adam']
    optimizer_histories = {
        'SGD': {'train_acc': [], 'val_acc': []},
        'RMSprop': {'train_acc': [], 'val_acc': []},
        'Adam': {'train_acc': [], 'val_acc': []}
    }

    # Train the model with different optimizers
    for optimizer_type in optimizers:
        print(f"Training with {optimizer_type} optimizer")
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            # Instantiate a new model for each run
            model = model_class()

            # Train the model
            history, early_stop_epoch, early_stop_history = train_model_rnn(
                model, train_loader, val_loader, num_epochs, device, model_name, optimizer_type=optimizer_type
            )

            # Record training and validation accuracy at early stopping point
            if early_stop_history:
                optimizer_histories[optimizer_type]['train_acc'].append(early_stop_history['train_acc'][-1])
                optimizer_histories[optimizer_type]['val_acc'].append(early_stop_history['val_acc'][-1])
            else:
                optimizer_histories[optimizer_type]['train_acc'].append(history['train_acc'][-1])
                optimizer_histories[optimizer_type]['val_acc'].append(history['val_acc'][-1])

    # Plot the training accuracy for each optimizer
    plt.figure(figsize=(12, 6))
    for optimizer_type in optimizers:
        plt.plot(range(1, num_runs + 1), optimizer_histories[optimizer_type]['train_acc'], label=f'{optimizer_type} - Train')
    plt.xlabel('Run')
    plt.ylabel('Training Accuracy (%)')
    plt.title('Training Accuracy for Different Optimizers')
    plt.legend()
    plt.show()

    # Plot the validation accuracy for each optimizer
    plt.figure(figsize=(12, 6))
    for optimizer_type in optimizers:
        plt.plot(range(1, num_runs + 1), optimizer_histories[optimizer_type]['val_acc'], label=f'{optimizer_type} - Validation')
    plt.xlabel('Run')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy for Different Optimizers')
    plt.legend()
    plt.show()

def train_model_multiple_learning_rates(model_class, train_loader, val_loader, device, model_name, num_runs=3, num_epochs=30):
    """
    Train the model multiple times with different learning rates and plot the training and validation accuracies.

    Args:
        model_class (type): The class of the model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to use for training (CPU or GPU).
        model_name (str): The name of the model (used for saving checkpoints).
        num_runs (int): Number of times to train the model per learning rate.
        num_epochs (int): Number of epochs to train the model in each run.
    """
    learning_rates = [0.001, 0.005, 0.01]
    learning_rate_histories = {
        0.001: {'train_acc': [], 'val_acc': []},
        0.005: {'train_acc': [], 'val_acc': []},
        0.01: {'train_acc': [], 'val_acc': []}
    }

    # Train the model with different learning rates
    for learning_rate in learning_rates:
        print(f"Training with learning rate {learning_rate}")
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            # Instantiate a new model for each run
            model = model_class()

            # Train the model
            history, early_stop_epoch, early_stop_history = train_model_rnn(
                model, train_loader, val_loader, num_epochs, device, model_name, learning_rate=learning_rate
            )

            # Record training and validation accuracy at early stopping point
            if early_stop_history:
                learning_rate_histories[learning_rate]['train_acc'].append(early_stop_history['train_acc'][-1])
                learning_rate_histories[learning_rate]['val_acc'].append(early_stop_history['val_acc'][-1])
            else:
                learning_rate_histories[learning_rate]['train_acc'].append(history['train_acc'][-1])
                learning_rate_histories[learning_rate]['val_acc'].append(history['val_acc'][-1])

    # Plot the training accuracy for each learning rate
    plt.figure(figsize=(12, 6))
    for learning_rate in learning_rates:
        plt.plot(range(1, num_runs + 1), learning_rate_histories[learning_rate]['train_acc'], label=f'LR {learning_rate} - Train')
    plt.xlabel('Run')
    plt.ylabel('Training Accuracy (%)')
    plt.title('Training Accuracy for Different Learning Rates')
    plt.legend()
    plt.show()

    # Plot the validation accuracy for each learning rate
    plt.figure(figsize=(12, 6))
    for learning_rate in learning_rates:
        plt.plot(range(1, num_runs + 1), learning_rate_histories[learning_rate]['val_acc'], label=f'LR {learning_rate} - Validation')
    plt.xlabel('Run')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy for Different Learning Rates')
    plt.legend()
    plt.show()

def train_model_multiple_batch_sizes(model_class, train_data, val_data, device, model_name, num_runs=3, num_epochs=30):
    """
    Train the model multiple times with different batch sizes and plot the training and validation accuracies.

    Args:
        model_class (type): The class of the model to be trained.
        train_data (Dataset): Dataset for training data.
        val_data (Dataset): Dataset for validation data.
        device (torch.device): Device to use for training (CPU or GPU).
        model_name (str): The name of the model (used for saving checkpoints).
        num_runs (int): Number of times to train the model per batch size.
        num_epochs (int): Number of epochs to train the model in each run.
    """
    batch_sizes = [16, 64, 32]
    batch_size_histories = {
        16: {'train_acc': [], 'val_acc': []},
        64: {'train_acc': [], 'val_acc': []},
        32: {'train_acc': [], 'val_acc': []}
    }

    # Train the model with different batch sizes
    for batch_size in batch_sizes:
        print(f"Training with batch size {batch_size}")
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            # Create DataLoader with the new batch size
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

            # Instantiate a new model for each run
            model = model_class()

            # Train the model
            history, early_stop_epoch, early_stop_history = train_model_rnn(
                model, train_loader, val_loader, num_epochs, device, model_name
            )

            # Record training and validation accuracy at early stopping point
            if early_stop_history:
                batch_size_histories[batch_size]['train_acc'].append(early_stop_history['train_acc'][-1])
                batch_size_histories[batch_size]['val_acc'].append(early_stop_history['val_acc'][-1])
            else:
                batch_size_histories[batch_size]['train_acc'].append(history['train_acc'][-1])
                batch_size_histories[batch_size]['val_acc'].append(history['val_acc'][-1])

    # Plot the training accuracy for each batch size
    plt.figure(figsize=(12, 6))
    for batch_size in batch_sizes:
        plt.plot(range(1, num_runs + 1), batch_size_histories[batch_size]['train_acc'], label=f'Batch Size {batch_size} - Train')
    plt.xlabel('Run')
    plt.ylabel('Training Accuracy (%)')
    plt.title('Training Accuracy for Different Batch Sizes')
    plt.legend()
    plt.show()

    # Plot the validation accuracy for each batch size
    plt.figure(figsize=(12, 6))
    for batch_size in batch_sizes:
        plt.plot(range(1, num_runs + 1), batch_size_histories[batch_size]['val_acc'], label=f'Batch Size {batch_size} - Validation')
    plt.xlabel('Run')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy for Different Batch Sizes')
    plt.legend()
    plt.show()
