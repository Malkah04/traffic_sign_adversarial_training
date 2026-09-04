from art.attacks.evasion import ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainer
from .basemodel import art_classifier, base_model
from .data import test_loader, train_loader
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier

x_train_list = []
y_train_list = []

for x, y in train_loader:
    x_train_list.append(x.numpy())
    y_train_list.append(y.numpy())

x_train = np.concatenate(x_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)

print("Training data:", x_train.shape)
print("Training labels:", y_train.shape)


x_test_list = []
y_test_list = []

print("heeeeee")

for x, y in test_loader:
    x_test_list.append(x.numpy())
    y_test_list.append(y.numpy())

x_test = np.concatenate(x_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

print("Testing data:", x_test.shape)
print("Testing labels:", y_test.shape)


defense_model = copy.deepcopy(base_model)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    defense_model.parameters(),
    lr=0.001
)

device_type = "gpu" if torch.cuda.is_available() else "cpu"

defense_classifier = PyTorchClassifier(
    model=defense_model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=art_classifier.input_shape,
    nb_classes=43,
    clip_values=(0, 1),
    device_type=device_type
)


pgd_train = ProjectedGradientDescent(
    estimator=defense_classifier,
    eps=0.01,
    eps_step=0.001,
    max_iter=4
)


adv_trainer = AdversarialTrainer(
    defense_classifier,
    attacks=pgd_train,
    ratio=1.0
)


print("\nStarting adversarial training...")


adv_trainer.fit(
    x_train,
    y_train,
    batch_size=512,
    nb_epochs=5
)


print("Adversarial training finished.")


pred_clean = defense_classifier.predict(x_test)

clean_accuracy = np.mean(np.argmax(pred_clean, axis=1) == y_test)


print(f"\nDefended model clean test accuracy: "f"{clean_accuracy * 100:.2f}%")


total_correct_pgd = 0
total_samples_pgd = 0

for x_batch, y_batch in test_loader:

    x_batch_numpy = x_batch.numpy()
    y_batch_numpy = y_batch.numpy()

    # Generate PGD adversarial examples
    x_adv = pgd_train.generate(x=x_batch_numpy)

    # Predict using the defended model
    predictions = defense_classifier.predict(x_adv)

    predicted_labels = np.argmax(predictions, axis=1)

    total_correct_pgd += np.sum(
        predicted_labels == y_batch_numpy
    )

    total_samples_pgd += len(y_batch_numpy)


pgd_accuracy = total_correct_pgd / total_samples_pgd


print(f"\nDefended model PGD test accuracy: {pgd_accuracy * 100:.2f}%")