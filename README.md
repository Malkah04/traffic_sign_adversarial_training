# Adversarial Attacks & Defenses on Traffic Sign Classification

A practical exploration of **Adversarial Machine Learning** using a CNN for traffic sign classification.

The project investigates how small, carefully crafted perturbations can reduce model accuracy and how **adversarial training** can improve robustness against these attacks.

---

## 📌 Project Overview

Deep learning models can achieve high accuracy on clean images while still being vulnerable to adversarial examples.

In this project, I implemented and evaluated:

- **FGSM (Fast Gradient Sign Method)**

- **PGD (Projected Gradient Descent)**

- **FGSM Adversarial Training**

- **PGD Adversarial Training**

The experiments were performed on an RGB traffic-sign image dataset with **43 classes**, using images resized to **32×32**.
---
## 🧠 Model

A simple Convolutional Neural Network (CNN) was used as the baseline classifier.

### Architecture
```text

Input: 3 × 32 × 32

↓

Conv2D (3 → 32) + ReLU

↓

MaxPooling

↓

Conv2D (32 → 64) + ReLU

↓

MaxPooling

↓

Flatten

↓

Fully Connected

↓

43 Classes

Training Configuration

Loss: Cross Entropy Loss

Optimizer: Adam

Learning Rate: 0.001

⚔️ Adversarial Attacks

1. FGSM

FGSM generates an adversarial image by taking one step in the direction that increases the model's loss.

xadv=x+ϵ⋅sign(∇xL)x_{adv} = x + \epsilon \cdot sign(\nabla_x L)

Where:



x = original image

L = model loss

∇xL = gradient of the loss with respect to the input

sign = direction of the gradient

ε = perturbation budget

The key idea is:



During normal training, gradients are used to update the model's weights.

During an adversarial attack, the gradient is used to modify the input.

📈 FGSM Epsilon Sweep

The attack strength was evaluated using different values of ε.

EpsilonApprox. Accuracy0.001~88%0.005~73%0.01~54%0.03~32%0.05~24%0.10~12%

Clean accuracy was approximately 89%.

At ε = 0.01, accuracy dropped to approximately 54%, showing the model's vulnerability to small perturbations.

2. PGD

PGD extends the idea of FGSM by applying multiple smaller gradient-based steps.

Instead of modifying the image once, PGD repeatedly:



Calculates the gradient with respect to the input.

Takes a small step in the direction that increases the loss.

Projects the perturbed image back into the allowed ε region.

For a fair comparison with FGSM, both attacks were evaluated using:



ε = 0.01

Results

AttackAccuracyClean89.37%FGSM53.81%PGD50.42%

PGD reduced the accuracy further than FGSM at the same perturbation budget.

🛡️ Adversarial Training

Adversarial training improves robustness by including adversarial examples during training.

Conceptually:



Clean Image

↓

Generate Adversarial Example

↓

Adversarial Image

↓

Train Model

The goal is to make the model perform well not only on clean inputs, but also on inputs specifically designed to fool it.

FGSM Adversarial Training

The baseline model was trained using FGSM-generated adversarial examples.



Results

ModelClean AccuracyFGSM AccuracyBaseline89.37%53.81%FGSM Defense92.41%76.40%

Robustness against FGSM improved by:

+22.59 percentage points

PGD Adversarial Training

PGD adversarial training was then applied using:



ε = 0.01

ε step = 0.001

Training PGD iterations = 4

Results

ModelClean AccuracyPGD AccuracyBaseline89.37%50.42%PGD Defense91.62%85.40%

Robustness against PGD improved by approximately:

+34.98 percentage points

Clean accuracy also improved from 89.37% → 91.62%.



Note: The PGD attack used during adversarial training used fewer iterations than the 40-step PGD attack used for the main evaluation. This was a practical trade-off to keep adversarial training computationally manageable.

📊 Main Results

ExperimentAccuracyBaseline — Clean89.37%Baseline — FGSM (ε=0.01)53.81%Baseline — PGD (ε=0.01)50.42%FGSM Defense — Clean92.41%FGSM Defense — FGSM76.40%PGD Defense — Clean91.62%PGD Defense — PGD85.40%

Key Observation

A model with high clean accuracy can still be highly vulnerable to adversarial inputs.

Adversarial training significantly improved robustness in these experiments.

🛠️ Technologies

Python

PyTorch

Torchvision

NumPy

Matplotlib

Adversarial Robustness Toolbox (ART)

The exact module names may differ depending on the final project structure.

📚 What This Project Demonstrates

This project provides practical experience with:



Gradient-based adversarial attacks

Input gradients and loss maximization

Perturbation budgets (ε)

FGSM vs. iterative PGD

Adversarial example generation

Adversarial training

Model robustness evaluation

PyTorch with the Adversarial Robustness Toolbox

🔬 Key Takeaway

High accuracy does not necessarily mean high robustness.

Testing a model only on clean data can hide significant vulnerabilities. Evaluating it against adversarial attacks provides a better understanding of how robust the model actually is.

👩‍💻 Author

Malak Sobhy

Machine Learning | ML Security | Trustworthy AI





### قبل ما تعمليه Push

في حاجتين بس أنصحك تعدليهم:



1. بدّلي:

```text

<YOUR_REPOSITORY_URL>

برابط الـrepo الحقيقي.



جزء Project Structure لازم نراجعه على أسماء الملفات الحقيقية عندك، عشان الـREADME ما يقولش حاجة مختلفة عن المشروع الفعلي.

والأهم: ما تحطيش أرقام تقريبية في جدول Main Results لأن دي أرقام الـresults الفعلية عندك؛ الـ~ نخليها فقط في الـepsilon sweep اللي الصورة فيها تقريبية. اعملهولي ف فايل وابعته

حفّظت لك الملف بالكامل باسم README.md مع إصلاح كل تنسيقات الماركداون (المعادلات والجداول)، وتدقيق صيغ النتائج لتكون دقيقة ومضبوطة.

يمكنك تحميل الملف جاهزاً مباشرة، أو نسخ محتواه من المربع أدناه:

Markdown
# Adversarial Attacks & Defenses on Traffic Sign Classification

A practical exploration of **Adversarial Machine Learning** using a CNN for traffic sign classification.

The project investigates how small, carefully crafted perturbations can reduce model accuracy and how **adversarial training** can improve robustness against these attacks.

---

## 📌 Project Overview

Deep learning models can achieve high accuracy on clean images while still being vulnerable to adversarial examples.

In this project, I implemented and evaluated:

- **FGSM (Fast Gradient Sign Method)**
- **PGD (Projected Gradient Descent)**
- **FGSM Adversarial Training**
- **PGD Adversarial Training**

The experiments were performed on an RGB traffic-sign image dataset with **43 classes**, using images resized to **32×32**.

---

## 🧠 Model

A simple Convolutional Neural Network (CNN) was used as the baseline classifier.

### Architecture

```text
Input: 3 × 32 × 32
        ↓
Conv2D (3 → 32) + ReLU
        ↓
MaxPooling
        ↓
Conv2D (32 → 64) + ReLU
        ↓
MaxPooling
        ↓
Flatten
        ↓
Fully Connected
        ↓
43 Classes
Training Configuration
Loss: Cross Entropy Loss

Optimizer: Adam

Learning Rate: 0.001

⚔️ Adversarial Attacks
1. FGSM
FGSM generates an adversarial image by taking one step in the direction that increases the model's loss.

x 
adv
​
 =x+ϵ⋅sign(∇ 
x
​
 L)
Where:

x = original image

L = model loss

∇ 
x
​
 L = gradient of the loss with respect to the input

sign = direction of the gradient

ϵ = perturbation budget

The key idea is:

During normal training, gradients are used to update the model's weights.

During an adversarial attack, the gradient is used to modify the input.

📈 FGSM Epsilon Sweep
The attack strength was evaluated using different values of ϵ.

Epsilon (ϵ)	Approx. Accuracy
0.001	~88%
0.005	~73%
0.01	~54%
0.03	~32%
0.05	~24%
0.10	~12%
Clean accuracy was approximately 89%. At ϵ=0.01, accuracy dropped to approximately 54%, showing the model's vulnerability to small perturbations.

2. PGD
PGD extends the idea of FGSM by applying multiple smaller gradient-based steps.

Instead of modifying the image once, PGD repeatedly:

Calculates the gradient with respect to the input.

Takes a small step in the direction that increases the loss.

Projects the perturbed image back into the allowed ϵ region.

For a fair comparison with FGSM, both attacks were evaluated using ϵ=0.01:

Attack	Accuracy
Clean	89.37%
FGSM	53.81%
PGD	50.42%
PGD reduced the accuracy further than FGSM at the same perturbation budget.

🛡️ Adversarial Training
Adversarial training improves robustness by including adversarial examples during training.

Conceptually:

Plaintext
Clean Image ──> Generate Adversarial Example ──> Adversarial Image ──> Train Model
The goal is to make the model perform well not only on clean inputs, but also on inputs specifically designed to fool it.

FGSM Adversarial Training
The baseline model was trained using FGSM-generated adversarial examples.

Model	Clean Accuracy	FGSM Accuracy
Baseline	89.37%	53.81%
FGSM Defense	92.41%	76.40%
Robustness against FGSM improved by +22.59 percentage points.

PGD Adversarial Training
PGD adversarial training was then applied using:

ϵ=0.01

ϵ step=0.001

Training PGD iterations = 4

Model	Clean Accuracy	PGD Accuracy
Baseline	89.37%	50.42%
PGD Defense	91.62%	85.40%
Robustness against PGD improved by +34.98 percentage points. Clean accuracy also improved from 89.37% → 91.62%.

Note: The PGD attack used during adversarial training used fewer iterations than the 40-step PGD attack used for the main evaluation. This was a practical trade-off to keep adversarial training computationally manageable.

📊 Main Results
Experiment	Accuracy
Baseline — Clean	89.37%
Baseline — FGSM (ϵ=0.01)	53.81%
Baseline — PGD (ϵ=0.01)	50.42%
FGSM Defense — Clean	92.41%
FGSM Defense — FGSM	76.40%
PGD Defense — Clean	91.62%
PGD Defense — PGD	85.40%
Key Observation
A model with high clean accuracy can still be highly vulnerable to adversarial inputs.

Adversarial training significantly improved robustness in these experiments.

🛠️ Technologies
Language: Python

Frameworks: PyTorch, Torchvision

Libraries: NumPy, Matplotlib, Adversarial Robustness Toolbox (ART)

📂 Project Structure
Plaintext
traffic-sign-adv-attack/
│
├── app/
│   ├── basemodel.py
│   ├── data.py
│   ├── fgsm.py
│   ├── pgd.py
│   ├── fgsm_defense.py
│   └── pgd_defense.py
│
├── images/
│   ├── original_vs_adversarial.png
│   ├── epsilon_sweep.png
│   └── results.png
│
├── requirements.txt
└── README.md
🚀 Installation
Clone the repository:

Bash
git clone [https://github.com/Malkah04/traffic-sign-adv-attack.git](https://github.com/Malkah04/traffic-sign-adv-attack.git)
cd traffic-sign-adv-attack
Create and activate a virtual environment:

Bash
python3 -m venv .venv
source .venv/bin/activate
Install dependencies:

Bash
pip install -r requirements.txt
▶️ Running the Project
Run the experiment modules from the project root using Python's module flag (-m):

Bash
# Evaluate FGSM and PGD Attacks
python -m app.fgsm
python -m app.pgd

# Train and evaluate Defense Models
python -m app.fgsm_defense
python -m app.pgd_defense
📚 What This Project Demonstrates
Gradient-based adversarial attacks

Input gradients and loss maximization

Perturbation budgets (ϵ)

FGSM vs. iterative PGD

Adversarial example generation & training

Model robustness evaluation

Implementing security workflows with PyTorch and ART

🔬 Key Takeaway
High accuracy does not necessarily mean high robustness. Testing a model only on clean data can hide significant vulnerabilities. Evaluating it against adversarial attacks provides a complete understanding of model reliability.

👩‍💻 Author
Malak Sobhy

Machine Learning | ML Security | Trustworthy AI

عايزاه بصيغه md

Training Configuration
Loss: Cross Entropy Loss

Optimizer: Adam

Learning Rate: 0.001

⚔️ Adversarial Attacks
1. FGSM
FGSM generates an adversarial image by taking one step in the direction that increases the model's loss.

x 
adv
​
 =x+ϵ⋅sign(∇ 
x
​
 L)
Where:

x = original image

L = model loss

∇ 
x
​
 L = gradient of the loss with respect to the input

sign = direction of the gradient

ϵ = perturbation budget

The key idea is:

During normal training, gradients are used to update the model's weights.

During an adversarial attack, the gradient is used to modify the input.

📈 FGSM Epsilon Sweep
The attack strength was evaluated using different values of ϵ.

Epsilon (ϵ)	Approx. Accuracy
0.001	~88%
0.005	~73%
0.01	~54%
0.03	~32%
0.05	~24%
0.10	~12%
Clean accuracy was approximately 89%. At ϵ=0.01, accuracy dropped to approximately 54%, showing the model's vulnerability to small perturbations.

2. PGD
PGD extends the idea of FGSM by applying multiple smaller gradient-based steps.

Instead of modifying the image once, PGD repeatedly:

Calculates the gradient with respect to the input.

Takes a small step in the direction that increases the loss.

Projects the perturbed image back into the allowed ϵ region.

For a fair comparison with FGSM, both attacks were evaluated using ϵ=0.01:

Attack	Accuracy
Clean	89.37%
FGSM	53.81%
PGD	50.42%
PGD reduced the accuracy further than FGSM at the same perturbation budget.

🛡️ Adversarial Training
Adversarial training improves robustness by including adversarial examples during training.

Conceptually:

Plaintext
Clean Image ──> Generate Adversarial Example ──> Adversarial Image ──> Train Model
The goal is to make the model perform well not only on clean inputs, but also on inputs specifically designed to fool it.

FGSM Adversarial Training
The baseline model was trained using FGSM-generated adversarial examples.

Model	Clean Accuracy	FGSM Accuracy
Baseline	89.37%	53.81%
FGSM Defense	92.41%	76.40%
Robustness against FGSM improved by +22.59 percentage points.

PGD Adversarial Training
PGD adversarial training was then applied using:

ϵ=0.01

ϵ step=0.001

Training PGD iterations = 4

Model	Clean Accuracy	PGD Accuracy
Baseline	89.37%	50.42%
PGD Defense	91.62%	85.40%
Robustness against PGD improved by +34.98 percentage points. Clean accuracy also improved from 89.37% → 91.62%.

Note: The PGD attack used during adversarial training used fewer iterations than the 40-step PGD attack used for the main evaluation. This was a practical trade-off to keep adversarial training computationally manageable.

📊 Main Results
Experiment	Accuracy
Baseline — Clean	89.37%
Baseline — FGSM (ϵ=0.01)	53.81%
Baseline — PGD (ϵ=0.01)	50.42%
FGSM Defense — Clean	92.41%
FGSM Defense — FGSM	76.40%
PGD Defense — Clean	91.62%
PGD Defense — PGD	85.40%
Key Observation
A model with high clean accuracy can still be highly vulnerable to adversarial inputs.

Adversarial training significantly improved robustness in these experiments.

🛠️ Technologies
Language: Python

Frameworks: PyTorch, Torchvision

Libraries: NumPy, Matplotlib, Adversarial Robustness Toolbox (ART)

📂 Project Structure
Plaintext
traffic-sign-adv-attack/
│
├── app/
│   ├── basemodel.py
│   ├── data.py
│   ├── fgsm.py
│   ├── pgd.py
│   ├── fgsm_defense.py
│   └── pgd_defense.py
│
├── images/
│   ├── original_vs_adversarial.png
│   ├── epsilon_sweep.png
│   └── results.png
│
├── requirements.txt
└── README.md
🚀 Installation
Clone the repository:

Bash
git clone [https://github.com/Malkah04/traffic-sign-adv-attack.git](https://github.com/Malkah04/traffic-sign-adv-attack.git)
cd traffic-sign-adv-attack
Create and activate a virtual environment:

Bash
python3 -m venv .venv
source .venv/bin/activate
Install dependencies:

Bash
pip install -r requirements.txt
▶️ Running the Project
Run the experiment modules from the project root using Python's module flag (-m):

Bash
# Evaluate FGSM and PGD Attacks
python -m app.fgsm
python -m app.pgd

# Train and evaluate Defense Models
python -m app.fgsm_defense
python -m app.pgd_defense
📚 What This Project Demonstrates
Gradient-based adversarial attacks

Input gradients and loss maximization

Perturbation budgets (ϵ)

FGSM vs. iterative PGD

Adversarial example generation & training

Model robustness evaluation

Implementing security workflows with PyTorch and ART

🔬 Key Takeaway
High accuracy does not necessarily mean high robustness. Testing a model only on clean data can hide significant vulnerabilities. Evaluating it against adversarial attacks provides a complete understanding of model reliability.

👩‍💻 Author
Malak Sobhy

Machine Learning | ML Security | Trustworthy AI
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(readme_content)

print("README.md file created successfully.")


```text?code_stdout&code_event_index=1
README.md file created successfully.

تم تجهيز ملف README.md بصيغة Markdown المطلوبة تماماً.

MD icon
README
 MD 
Gemini is AI and can make mistakes.




Analyzing
# Adversarial Attacks & Defenses on Traffic Sign Classification

A practical exploration of **Adversarial Machine Learning** using a CNN for traffic sign classification.

The project investigates how small, carefully crafted perturbations can reduce model accuracy and how **adversarial training** can improve robustness against these attacks.

---

## 📌 Project Overview

Deep learning models can achieve high accuracy on clean images while still being vulnerable to adversarial examples.

In this project, I implemented and evaluated:

- **FGSM (Fast Gradient Sign Method)**
- **PGD (Projected Gradient Descent)**
- **FGSM Adversarial Training**
- **PGD Adversarial Training**

The experiments were performed on an RGB traffic-sign image dataset with **43 classes**, using images resized to **32×32**.

---

## 🧠 Model

A simple Convolutional Neural Network (CNN) was used as the baseline classifier.

### Architecture

```text
Input: 3 × 32 × 32
        ↓
Conv2D (3 → 32) + ReLU
        ↓
MaxPooling
        ↓
Conv2D (32 → 64) + ReLU
        ↓
MaxPooling
        ↓
Flatten
        ↓
Fully Connected
        ↓
43 Classes
```

### Training Configuration
- **Loss:** Cross Entropy Loss
- **Optimizer:** Adam
- **Learning Rate:** 0.001

---

## ⚔️ Adversarial Attacks

### 1. FGSM
FGSM generates an adversarial image by taking one step in the direction that increases the model's loss.

$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L)$$

**Where:**
- $x$ = original image
- $L$ = model loss
- $\nabla_x L$ = gradient of the loss with respect to the input
- $\text{sign}$ = direction of the gradient
- $\epsilon$ = perturbation budget

**The key idea is:**
- During normal training, gradients are used to update the model's weights.
- During an adversarial attack, the gradient is used to modify the input.

#### 📈 FGSM Epsilon Sweep
The attack strength was evaluated using different values of $\epsilon$.

| Epsilon ($\epsilon$) | Approx. Accuracy |
| :--- | :--- |
| **0.001** | ~88% |
| **0.005** | ~73% |
| **0.01** | ~54% |
| **0.03** | ~32% |
| **0.05** | ~24% |
| **0.10** | ~12% |

*Clean accuracy was approximately 89%. At $\epsilon = 0.01$, accuracy dropped to approximately 54%, showing the model's vulnerability to small perturbations.*

### 2. PGD
PGD extends the idea of FGSM by applying multiple smaller gradient-based steps.

Instead of modifying the image once, PGD repeatedly:
1. Calculates the gradient with respect to the input.
2. Takes a small step in the direction that increases the loss.
3. Projects the perturbed image back into the allowed $\epsilon$ region.

For a fair comparison with FGSM, both attacks were evaluated using $\epsilon = 0.01$:

| Attack | Accuracy |
| :--- | :--- |
| **Clean** | 89.37% |
| **FGSM** | 53.81% |
| **PGD** | 50.42% |

*PGD reduced the accuracy further than FGSM at the same perturbation budget.*

---

## 🛡️ Adversarial Training

Adversarial training improves robustness by including adversarial examples during training.

**Conceptually:**
```text
Clean Image ──> Generate Adversarial Example ──> Adversarial Image ──> Train Model
```

The goal is to make the model perform well not only on clean inputs, but also on inputs specifically designed to fool it.

### FGSM Adversarial Training
The baseline model was trained using FGSM-generated adversarial examples.

| Model | Clean Accuracy | FGSM Accuracy |
| :--- | :--- | :--- |
| **Baseline** | 89.37% | 53.81% |
| **FGSM Defense** | 92.41% | 76.40% |

*Robustness against FGSM improved by **+22.59 percentage points**.*

### PGD Adversarial Training
PGD adversarial training was then applied using:
- $\epsilon = 0.01$
- $\epsilon \text{ step} = 0.001$
- Training PGD iterations = 4

| Model | Clean Accuracy | PGD Accuracy |
| :--- | :--- | :--- |
| **Baseline** | 89.37% | 50.42% |
| **PGD Defense** | 91.62% | 85.40% |

*Robustness against PGD improved by **+34.98 percentage points**. Clean accuracy also improved from 89.37% → 91.62%.*

> **Note:** The PGD attack used during adversarial training used fewer iterations than the 40-step PGD attack used for the main evaluation. This was a practical trade-off to keep adversarial training computationally manageable.

---

## 📊 Main Results

| Experiment | Accuracy |
| :--- | :--- |
| **Baseline — Clean** | 89.37% |
| **Baseline — FGSM ($\epsilon=0.01$)** | 53.81% |
| **Baseline — PGD ($\epsilon=0.01$)** | 50.42% |
| **FGSM Defense — Clean** | 92.41% |
| **FGSM Defense — FGSM** | 76.40% |
| **PGD Defense — Clean** | 91.62% |
| **PGD Defense — PGD** | 85.40% |

### Key Observation
- A model with high clean accuracy can still be highly vulnerable to adversarial inputs.
- Adversarial training significantly improved robustness in these experiments.

---

## 🛠️ Technologies

- **Language:** Python
- **Frameworks:** PyTorch, Torchvision
- **Libraries:** NumPy, Matplotlib, Adversarial Robustness Toolbox (ART)

---

## 📚 What This Project Demonstrates

- Gradient-based adversarial attacks
- Input gradients and loss maximization
- Perturbation budgets ($\epsilon$)
- FGSM vs. iterative PGD
- Adversarial example generation & training
- Model robustness evaluation
- Implementing security workflows with PyTorch and ART

---

## 🔬 Key Takeaway

High accuracy does not necessarily mean high robustness. Testing a model only on clean data can hide significant vulnerabilities. Evaluating it against adversarial attacks provides a complete understanding of model reliability.

---

## 👩‍💻 Author

**Malak Sobhy**  
*Machine Learning | ML Security | Trustworthy AI*
