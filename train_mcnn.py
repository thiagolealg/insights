
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

# --- MCNN Model ---
class TradeMCNN(nn.Module):
    def __init__(self):
        super(TradeMCNN, self).__init__()
        
        # Input: (Batch, 1, 5, 11) - (Channels, Height=Features, Width=Time)
        
        # Conv Block 1
        # Kernel (1, 2) -> Slides over time, independent rows (mostly)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), padding=(0,0))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2)) # Reduce time dim
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 2), padding=(0,0))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        
        # After pools, what is the size?
        # W=11 -> Conv(1,2) -> W=10 -> Pool(1,2) -> W=5
        # W=5 -> Conv(1,2) -> W=4 -> Pool(1,2) -> W=2
        # Height=5 -> Unchanged by (1,x) kernels? 
        # Yes, height stays 5.
        # So Final Map: (Batch, 64, 5, 2)
        # Flatten: 64 * 5 * 2 = 640
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(64 * 5 * 2, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        self.out = nn.Linear(128, 3) # 0=Hold, 1=Buy, 2=Sell

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        return self.out(x)

def train_model():
    print("--- Training MCNN (Paper Replication) ---")
    
    # Load Data
    try:
        X = torch.load('train_inputs_mcnn.pt')
        y = torch.load('train_labels_mcnn.pt')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Data Loaded: X={X.shape}, y={y.shape}")
    # X shape: (N, 1, 5, 11)
    
    # Validation Split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Class Weights (if imbalanced)
    # 0=Hold, 1=Buy, 2=Sell
    # Usually Hold is dominant, but we balanced in generator.
    # Check counts
    unique, counts = torch.unique(y_train, return_counts=True)
    print(f"Class Counts: {dict(zip(unique.tolist(), counts.tolist()))}")
    
    # Dataset
    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)
    
    BATCH_SIZE = 64
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    model = TradeMCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    EPOCHS = 50
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "trade_mcnn.pth")
            print(f"  -> Model Saved (Acc: {best_acc:.2f}%)")
            
    print("Training Complete.")

if __name__ == "__main__":
    train_model()
