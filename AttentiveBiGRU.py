import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # Linear layer to calculate attention weights
        self.attention = nn.Linear(hidden_dim * 2, 1)  # *2 because GRU is bidirectional

    def forward(self, gru_output):
        # gru_output shape: (batch_size, seq_length, hidden_dim * 2)
        attention_weights = torch.tanh(self.attention(gru_output))
        attention_weights = torch.softmax(attention_weights, dim=1)  # Normalize over sequence length
        attended_output = torch.sum(attention_weights * gru_output, dim=1)
        return attended_output


class AttentiveBiGRUModel(nn.Module):
    config = {
        'input_dim': 300,
        'hidden_dim': 512,
        'num_layers': 3,
        'num_classes': 2,
        'dropout': 0.5
    }
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.batch_norm_input = nn.BatchNorm1d(input_dim)

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = Attention(hidden_dim)

        self.batch_norm1 = nn.BatchNorm1d(hidden_dim * 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        x_reshaped = x.reshape(-1, x.shape[-1])
        x = self.batch_norm_input(x_reshaped)
        x = x.reshape(batch_size, -1, x.shape[-1])

        out, hidden = self.gru(x)

        # Apply attention mechanism on the output sequence from GRU
        out = self.attention(out)

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch_norm2(out)
        out = self.dropout2(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)

        return out
