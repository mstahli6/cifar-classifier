import torch
import torch.nn as nn
import torch.nn.functional as F


# src/model.py
class GeneralCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, dropout_rate=0.3, 
                 num_blocks=3, base_filters=32, fc1_size=256): # <--- Check these!
        super(GeneralCNN, self).__init__()
        
        self.layers = nn.ModuleList()
        curr_in = in_channels
        curr_out = base_filters
        
        for i in range(num_blocks):
            self.layers.append(nn.Conv2d(curr_in, curr_out, 3, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2, 2))
            curr_in = curr_out
            curr_out *= 2 
            
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(dropout_rate)
        
        # curr_in is now the number of channels from the LAST conv layer
        self.fc1 = nn.Linear(curr_in * 4 * 4, fc1_size)
        self.fc2 = nn.Linear(fc1_size, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)