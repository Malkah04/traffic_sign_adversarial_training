# Adversarial Attacks & Defenses on Traffic Sign Classification

## 📌 Project Overview

This project explores **Adversarial Machine Learning** and the vulnerability of image classification models to carefully crafted adversarial perturbations.

A CNN was trained to classify traffic signs into **43 classes**. The model was then evaluated against adversarial attacks and trained with adversarial examples to improve its robustness.

---

## 🎯 Project Objectives

The main objectives of this project were to:

- Understand how adversarial examples affect image classification models.
- Implement **FGSM** and **PGD** attacks.
- Study the effect of different perturbation budgets ($\epsilon$).
- Evaluate the model's robustness against adversarial inputs.
- Apply **Adversarial Training** as a defense mechanism.
- Compare the model's performance before and after adversarial training.

---

## ⚔️ FGSM Attack

**Fast Gradient Sign Method (FGSM)** generates adversarial examples by modifying the input image in the direction that increases the model's loss.

The FGSM equation is:

$$
x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(\theta, x, y))
$$

Where:

- $x$ = original input image
- $x_{adv}$ = adversarial image
- $\epsilon$ = perturbation budget
- $L$ = loss function
- $\nabla_x L$ = gradient of the loss with respect to the input image
- $\text{sign}(\cdot)$ = sign of the gradient
- $\theta$ = model parameters
- $y$ = true label

The gradient is used to determine **how the input should be changed to increase the model's loss**, while $\epsilon$ controls the maximum perturbation strength.
