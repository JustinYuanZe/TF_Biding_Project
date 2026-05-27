import torch
import torch.nn as nn

class Conv1DBranch(nn.Module):
    """
    A single branch of the mCNN with Conv1d -> BatchNorm1d -> ReLU -> GlobalMaxPool1d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Conv1DBranch, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1) # Extract position-invariant maximum signal

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x.squeeze(-1) # Shape: (batch_size, out_channels)


class MultiScaleCNN(nn.Module):
    """
    Multi-Scale Convolutional Neural Network (mCNN) for DNA sequences.
    Takes DNABERT-2 embeddings of shape (batch_size, seq_len, embedding_dim)
    and applies parallel convolutions of different scale widths to detect motifs.
    """
    def __init__(self, embedding_dim=768, branch_channels=128, kernel_sizes=[3, 5, 7, 9], num_classes=4, dropout_rate=0.5):
        super(MultiScaleCNN, self).__init__()
        
        self.branches = nn.ModuleList([
            Conv1DBranch(
                in_channels=embedding_dim,
                out_channels=branch_channels,
                kernel_size=k,
                padding=k // 2 # Keeps spatial dimensions aligned if needed (though we pool at the end)
            ) for k in kernel_sizes
        ])
        
        # Concatenated feature size = number of branches * branch channels
        concatenated_dim = len(kernel_sizes) * branch_channels
        
        # Classification head
        self.fc_head = nn.Sequential(
            nn.Linear(concatenated_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len, embedding_dim)
        # Conv1d expects shape: (batch_size, embedding_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Forward through parallel branches
        branch_outputs = [branch(x) for branch in self.branches]
        
        # Concatenate outputs along the channel dimension
        feat = torch.cat(branch_outputs, dim=1) # Shape: (batch_size, concatenated_dim)
        
        # Classification prediction logits
        out = self.fc_head(feat)
        return out


class ImprovedOneHotCNN(nn.Module):
    """
    Highly regularized CNN for raw DNA sequences to prevent overfitting on small datasets.
    Key features:
    1. Learnable Embedding Layer: maps 4 base categories to a continuous 32-dim space.
    2. Conv1D branches with internal dropout to prevent co-adaptation of filters.
    3. Global MAX + AVG Pooling: MAX captures motif presence; AVG captures motif abundance/frequency.
    4. Small model capacity (~85K parameters) to avoid memorizing small training sets.
    """
    def __init__(self, seq_len=101, num_classes=4, embedding_dim=32, branch_channels=64, kernel_sizes=[3, 5, 7, 9], dropout_rate=0.6):
        super(ImprovedOneHotCNN, self).__init__()
        # Categories: A=0, C=1, G=2, T=3, N/Padding=4
        self.embedding = nn.Embedding(5, embedding_dim, padding_idx=4)
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=branch_channels,
                    kernel_size=k,
                    padding=k // 2
                ),
                nn.BatchNorm1d(branch_channels),
                nn.ReLU(),
                nn.Dropout(p=0.2) # Regularize intermediate feature maps
            ) for k in kernel_sizes
        ])
        
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Concat feature size = number of branches * branch channels * 2 (max + avg pools)
        concatenated_dim = len(kernel_sizes) * branch_channels * 2
        
        self.fc_head = nn.Sequential(
            nn.Linear(concatenated_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len) -> integer indices
        x = self.embedding(x)  # Shape: (batch_size, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # Shape: (batch_size, embedding_dim, seq_len)
        
        branch_feats = []
        for branch in self.branches:
            feat = branch(x)
            max_feat = self.max_pool(feat).squeeze(-1)
            avg_feat = self.avg_pool(feat).squeeze(-1)
            branch_feats.extend([max_feat, avg_feat])
            
        feat = torch.cat(branch_feats, dim=1) # Shape: (batch_size, concatenated_dim)
        out = self.fc_head(feat)
        return out

