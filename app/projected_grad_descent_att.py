from art.attacks.evasion import ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainer
from .basemodel import art_classifier, base_model
from .data import test_loader, train_loader
import numpy as np
import matplotlib as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier


pgd_att = ProjectedGradientDescent(
    estimator=art_classifier,
    eps=0.01,
    eps_step=0.001,
    max_iter=40
)

total_correct = 0
total_samples = 0
max_perturbation = 0
total_perturbation = 0
total_pixels = 0

# Save first batch for visualization
x_test_numpy = None
y_test = None
x_adv_visual = None
pred_pgd = None


for batch_idx, (x_batch, y_batch) in enumerate(test_loader):

    x_batch_numpy = x_batch.numpy()
    y_batch_numpy = y_batch.numpy()

    x_adv = pgd_att.generate(x=x_batch_numpy)

    predictions = art_classifier.predict(x_adv)
    predicted_labels = np.argmax(predictions, axis=1)

    total_correct += np.sum(predicted_labels == y_batch_numpy)
    total_samples += len(y_batch_numpy)

    perturbation = x_adv - x_batch_numpy

    max_perturbation = max(
        max_perturbation,
        np.max(np.abs(perturbation))
    )

    total_perturbation += np.sum(np.abs(perturbation))
    total_pixels += perturbation.size

    if batch_idx == 0:
        x_test_numpy = x_batch_numpy
        y_test = y_batch_numpy
        x_adv_visual = x_adv
        pred_pgd = predictions


accuracy_pgd = total_correct / total_samples

mean_perturbation = total_perturbation / total_pixels

print(f"Max perturbation: {max_perturbation}")
print(f"Mean perturbation: {mean_perturbation}")
print(f"PGD Accuracy: {accuracy_pgd * 100:.2f}%")


print(
    f"Accuracy on PGD adversarial examples "
    f"(ε={pgd_att.eps}): {accuracy_pgd * 100:.2f}%"
)

