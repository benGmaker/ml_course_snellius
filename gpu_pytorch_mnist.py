import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a directory for the MNIST dataset
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Define transformations (convert to tensor and normalize)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)

# Create DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print dataset stats
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# Instantiate and move model to device
model = SimpleNN().to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop (1 epoch for demonstration)
print("Starting training...")
for batch_idx, (data, target) in enumerate(train_loader):
    # Move data and target to device
    data, target = data.to(device), target.to(device)

    # Forward pass
    output = model(data)

    # Compute loss
    loss = criterion(output, target)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % 100 == 0:  # Print progress every 100 batches
        print(f"Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.4f}")

print("Training complete!")

# Test a single batch on GPU
print("Testing on a single batch...")
data, target = next(iter(test_loader))
data, target = data.to(device), target.to(device)

# Perform a forward pass
output = model(data)
predicted = torch.argmax(output, dim=1)

# Print results
print(f"Predicted labels: {predicted[:10].cpu().numpy()}")
print(f"True labels: {target[:10].cpu().numpy()}")