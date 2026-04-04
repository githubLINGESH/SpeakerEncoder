import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        
    def forward(self, x):
        # Squeeze
        squeeze = x.mean(dim=2)  # Global average pooling over time
        # Excitation
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        # Scale
        return x * excitation.unsqueeze(2)

class Res2NetBlock(nn.Module):
    """Res2Net block for multi-scale feature extraction"""
    def __init__(self, channels, kernel_size=5, dilation=1, scale=4):
        super().__init__()
        self.scale = scale
        self.channels = channels
        self.width = channels // scale
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(channels)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size, 
                      padding=dilation * (kernel_size - 1) // 2, 
                      dilation=dilation)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale - 1)])
        
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.se = SEBlock(channels)
        
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Split into groups
        xs = torch.split(x, self.width, dim=1)
        ys = []
        y = xs[0]
        for i in range(1, self.scale):
            y = self.relu(self.bns[i-1](self.convs[i-1](y + xs[i])))
            ys.append(y)
        ys = [xs[0]] + ys
        x = torch.cat(ys, dim=1)
        
        x = self.bn3(self.conv3(x))
        x = self.se(x)
        x = self.relu(x + residual)
        return x

class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN Speaker Encoder for Timbre Extraction
    Based on: "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation"
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config.TIMBRE_EMBED_DIM
        
        # Initial convolution
        self.conv1 = nn.Conv1d(80, 512, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(512, 512, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(512)
        
        # Res2Net blocks with dilations
        self.block1 = Res2NetBlock(512, kernel_size=5, dilation=1, scale=4)
        self.block2 = Res2NetBlock(512, kernel_size=5, dilation=2, scale=4)
        self.block3 = Res2NetBlock(512, kernel_size=5, dilation=3, scale=4)
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
        # Statistics pooling
        self.stats_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection
        self.fc = nn.Linear(512 * 3, self.embedding_dim)
        self.bn_out = nn.BatchNorm1d(self.embedding_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, frames, features) - log-mel spectrogram
        Returns:
            (batch, embedding_dim) - timbre embedding
        """
        # Transpose for Conv1d (batch, features, frames)
        x = x.transpose(1, 2)
        
        # Initial conv layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Res2Net blocks
        x = self.block1(x) + x
        x = self.block2(x) + x
        x = self.block3(x) + x
        
        # Attention pooling
        att_weights = self.attention(x)
        att_pool = torch.sum(x * att_weights, dim=2)
        
        # Statistics pooling
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        
        # Combine attention and statistics
        stats = torch.cat([mean, std, att_pool], dim=1)
        
        # Final projection
        embedding = self.fc(stats)
        embedding = self.bn_out(embedding)
        
        return embedding