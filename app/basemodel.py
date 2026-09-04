import torch
import torch.nn as nn
import torch.optim as optim
from .data import get_dataloaders ,train_loader ,test_loader
from sklearn.metrics import confusion_matrix
from art.estimators.classification import PyTorchClassifier


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2) # 32 *32 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2) # 16*16
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 43) # 8*8

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        return x

def train_model(model, train_loader, epochs=7):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    print("Training finished!")


def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy



def plot_confusion_matrix(model, test_loader, device="cpu"):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    cm = confusion_matrix(all_targets, all_preds)

    return cm


train_loader, test_loader = get_dataloaders()

base_model = SimpleCNN()
train_model(model= base_model ,train_loader=train_loader ,epochs=10 )
base_model.eval()
acc =evaluate_model(base_model , test_loader)
print(f"Accuracy on clean test data: {acc:.2f}%")
print(plot_confusion_matrix(base_model, test_loader))


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(base_model.parameters(), lr=0.001)

art_classifier = PyTorchClassifier(
    model=base_model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32), 
    nb_classes=43,          
    clip_values=(0, 1)      
)

print("ART classifier created successfully!")


