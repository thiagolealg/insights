import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# CONFIG
BATCH_SIZE = 4096 # Large batch for MLP
LEARNING_RATE = 0.001
EPOCHS = 100

class TradeMLP(nn.Module):
    def __init__(self):
        super(TradeMLP, self).__init__()
        # Input: (Batch, 7)
        self.fc1 = nn.Linear(7, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.LeakyReLU(0.1)
        
        self.out = nn.Linear(32, 2)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.out(x)
        return x

def train():
    print("Loading dataset.pt (MLP Inside Bar)...")
    try:
        X = torch.load('train_inputs.pt', weights_only=False)
        y = torch.load('train_labels.pt', weights_only=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Dataset Loaded. X: {X.shape}, y: {y.shape}")
    
    # Convert from Numpy to Tensor if needed
    if not torch.is_tensor(X):
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
    
    # Check shape
    if len(X.shape) != 2 or X.shape[1] != 7:
        print(f"ERROR: Expected (N, 7), got {X.shape}. Re-generate dataset (MLP Mode)!")
        return

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"TRAINING ON DEVICE: {device}")
    model = TradeMLP().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    print("Starting Training (MLP - Inside Bar)...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(dataloader)
        acc = 100 * correct / total
        
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {acc:.2f}% | LR: {current_lr:.6f}")
        
    torch.save(model.state_dict(), "trade_mlp.pth")
    print("Model saved to trade_mlp.pth")

if __name__ == "__main__":
    train()
