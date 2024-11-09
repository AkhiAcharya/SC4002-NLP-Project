import torch.nn.functional as F
import torch.nn as nn
import torch

class CNNModel(nn.Module):
    config = {
        'input_dim': 300,
        'num_filters': 128,
        'filter_sizes': [3, 4, 5],
        'num_classes': 2,
        'dropout': 0.5
    }
    def __init__(self, input_dim, num_filters, filter_sizes, num_classes, dropout=0.5):
        super().__init__()
        self.num_filters = num_filters

        self.batch_norm_input = nn.BatchNorm1d(input_dim)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, num_filters, k),
                nn.ReLU(),
                nn.BatchNorm1d(num_filters)
            ) for k in filter_sizes
        ])

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        total_filters = num_filters * len(filter_sizes)
        self.fc1 = nn.Linear(total_filters, total_filters // 2)
        self.fc2 = nn.Linear(total_filters // 2, num_classes)

        self.batch_norm1 = nn.BatchNorm1d(total_filters // 2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)

        conv_results = []
        for conv in self.convs:
            conv_out = conv(x)
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)
            conv_results.append(pooled)

        x = torch.cat(conv_results, dim=1)

        x = self.dropout1(x)


        x = self.fc1(x)
        x = self.relu(x)
        x = self.batch_norm1(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x