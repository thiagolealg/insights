import torch
import torch.nn as nn
import numpy as np
import os

# MODEL DEFINITION (Must match training)
class TradeCNN(nn.Module):
    def __init__(self):
        super(TradeCNN, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool1d(2) 
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool1d(2) 
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.LeakyReLU(0.1)
        self.pool3 = nn.MaxPool1d(2) 
        self.fc1 = nn.Linear(128, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.relu_fc1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(0.2) 
        self.fc2 = nn.Linear(64, 2) 

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.mean(dim=2) 
        x = self.dropout1(self.relu_fc1(self.bn_fc1(self.fc1(x))))
        x = self.fc2(x)
        return x

def debug_conf():
    model = TradeCNN()
    model.load_state_dict(torch.load("trade_cnn.pth", weights_only=False))
    model.eval()
    
    # Load Data
    X = torch.load("train_inputs.pt", weights_only=False)
    y = torch.load("train_labels.pt", weights_only=False)
    
    print(f"Loaded {len(X)} samples.")
    
    # Predict first 100
    subset = torch.tensor(X[:100])
    
    with torch.no_grad():
        outputs = model(subset)
        probs = torch.softmax(outputs, dim=1)
        
    confs, classes = torch.max(probs, 1)
    
    print("--- Confidence Stats (First 100) ---")
    print(f"Avg Conf: {confs.mean():.4f}")
    print(f"Max Conf: {confs.max():.4f}")
    print(f"Min Conf: {confs.min():.4f}")
    print(f"Pred Class 1 (Win) Count: {torch.sum(classes == 1)}")
    print(f"Pred Class 0 (Loss) Count: {torch.sum(classes == 0)}")
    
    # Check High Conf
    high_conf = (confs > 0.60).sum()
    print(f"Samples > 0.60 Conf: {high_conf}")

if __name__ == "__main__":
    debug_conf()
