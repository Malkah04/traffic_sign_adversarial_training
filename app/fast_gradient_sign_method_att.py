import numpy as np
import matplotlib.pyplot as plt  
from art.attacks.evasion import FastGradientMethod
from .basemodel import art_classifier, base_model ,SimpleCNN
from .data import test_loader ,train_loader
import torch
import torch.nn as nn
import torch.optim as optim
import os

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

print("ttttttesssrttttt")
x, y = next(iter(train_loader))

print(x.min())
print(x.max())
print(len(train_loader.dataset.classes))



epsilons = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]

results = {}

for e in epsilons:

    fgsm = FastGradientMethod(estimator=art_classifier,eps=e)
    total_correct = 0
    total_samples = 0
    max_perturbation = 0
    total_perturbation = 0
    total_pixels = 0

    for x_batch, y_batch in test_loader:
        x_batch_numpy = x_batch.numpy()
        y_batch_numpy = y_batch.numpy()
        x_adv_fgsm = fgsm.generate(x=x_batch_numpy)

        predictions = art_classifier.predict(x_adv_fgsm)
        predicted_labels = np.argmax(predictions, axis=1)

        total_correct += np.sum( predicted_labels == y_batch_numpy)
        total_samples += len(y_batch_numpy)

        perturbation = x_adv_fgsm - x_batch_numpy

        max_perturbation = max(max_perturbation,np.max(np.abs(perturbation)))

        total_perturbation += np.sum(np.abs(perturbation))

        total_pixels += perturbation.size

    accuracy = total_correct / total_samples

    mean_perturbation = total_perturbation / total_pixels

    results[e] = accuracy

    print(f"\nε = {e}")
    print(f"Max perturbation: {max_perturbation}")
    print(f"Mean perturbation: {mean_perturbation}")
    print(f"FGSM Accuracy: {accuracy * 100:.2f}%")


#####  visiulization model 
epsilon = 0.01

fgsm = FastGradientMethod(
    estimator=art_classifier,
    eps=epsilon
)


x_batch, y_batch = next(iter(test_loader))

x_batch_numpy = x_batch.numpy()
y_batch_numpy = y_batch.numpy()


x_adv_fgsm = fgsm.generate(x=x_batch_numpy)

predictions_fgsm = art_classifier.predict(x_adv_fgsm)

save_dir = "fgsm_visualizations"

os.makedirs(save_dir,exist_ok=True)

for i in range(len(x_batch)):
    original_image = x_batch_numpy[i]
    adversarial_image = x_adv_fgsm[i]
    true_label = y_batch_numpy[i]

    original_prediction = np.argmax(art_classifier.predict(original_image[np.newaxis, ...]),axis=1)[0]

    adversarial_prediction = np.argmax(predictions_fgsm[i])

    perturbation = (adversarial_image - original_image)


    original_plot = np.transpose(original_image,(1, 2, 0))

    adversarial_plot = np.transpose(adversarial_image,(1, 2, 0))

    perturbation_plot = np.transpose(perturbation,(1, 2, 0))
    perturbation_visual = np.clip((perturbation_plot + epsilon) / (2 * epsilon),0,1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)

    plt.imshow(np.clip(original_plot,0,1))

    plt.title(f"Original Image\n"f"True: {true_label} | "f"Pred: {original_prediction}")
    plt.axis("off")


    plt.subplot(1, 3, 2)
    plt.imshow(perturbation_visual)

    plt.title(f"FGSM Perturbation\n"f"ε = {epsilon}")

    plt.axis("off")
    plt.subplot(1, 3, 3)

    plt.imshow(np.clip(adversarial_plot,0,1))

    plt.title(f"Adversarial Image\n"f"Pred: {adversarial_prediction}")

    plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir,f"fgsm_example_{i + 1}_eps_{epsilon}.png")

    plt.savefig(save_path,dpi=300,bbox_inches="tight")

    plt.close()

    print(f"Saved: {save_path}")

print("\nFGSM experiment completed!")

print(f"Visualization images saved in: "f"{save_dir}/")

print("Accuracy plot saved as: ""fgsm_accuracy_vs_epsilon.png")

