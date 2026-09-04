from .basemodel import art_classifier, base_model ,SimpleCNN
from .data import test_loader ,train_loader
import torch
import torch.nn as nn
import torch.optim as optim

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod


defense_model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(defense_model.parameters(), lr=0.001)

defense_classifier = PyTorchClassifier(
    model=defense_model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 32, 32), 
    nb_classes=43,          
    clip_values=(0, 1)      
)

print("Defense ART classifier created successfully!")

fgsm = FastGradientMethod(estimator =defense_classifier ,eps=0.01)

epn =10
for ep in range(epn):
    defense_model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for x , y in train_loader:
        x_numpy =x.numpy()
        y_numpy =y.numpy()

        x_adv = fgsm.generate(x=x_numpy ,y=y_numpy)
        x_adv = torch.from_numpy(x_adv).float()

        optimizer.zero_grad()
        output=defense_model(x_adv)
        loss =criterion(output ,y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        predictions = torch.argmax(output, dim=1)
        total_correct+=(predictions ==y).sum().item()
        total_samples += y.size(0)
    epoch_loss = total_loss / total_samples
    epoch_accuracy = total_correct / total_samples

    print(
        f"Epoch [{ep + 1}/{epn}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_accuracy * 100:.2f}%"
    )

print("Adversarial training completed!")

defense_model.eval()

total_correct = 0
total_samples = 0

with torch.no_grad():
    for x, y in test_loader:

        output = defense_model(x)

        predictions = torch.argmax(output, dim=1)

        total_correct += (predictions == y).sum().item()
        total_samples += y.size(0)

clean_accuracy = total_correct / total_samples

print(f"Defense Model - Clean Accuracy: {clean_accuracy * 100:.2f}%")


fgsm = FastGradientMethod(estimator=defense_classifier,eps=0.01)
total_correct = 0
total_samples = 0

for x, y in test_loader:
    x_numpy = x.numpy()
    y_numpy = y.numpy()

    x_adv = fgsm.generate(
        x=x_numpy,
        y=y_numpy
    )

    x_adv = torch.from_numpy(x_adv).float()
    with torch.no_grad():
        output = defense_model(x_adv)

    predictions = torch.argmax(output, dim=1)

    total_correct += (predictions == y).sum().item()
    total_samples += y.size(0)


fgsm_accuracy = total_correct / total_samples

print(f"Defense Model - FGSM Accuracy (eps=0.01): "f"{fgsm_accuracy * 100:.2f}%")


