# Lab 4.4 — Attention Mechanism (Intro)

## 📌 Project Overview

This project demonstrates a basic Attention Mechanism using PyTorch. It builds a sequence classification model that combines an embedding layer, a GRU (Gated Recurrent Unit), an attention mechanism, and a fully connected layer.

The model learns to classify integer sequences based on whether the sum of their elements is even or odd. It also visualizes attention weights to show how the model distributes its focus across sequence positions.

## 🎯 Objectives

- Generate a toy dataset using PyTorch.
- Understand word/token embeddings for sequence data.
- Implement a GRU-based neural network.
- Calculate attention scores and attention weights.
- Train a sequence classification model.
- Visualize attention weights using Matplotlib.

## 🛠️ Technologies Used

- **Python** — Programming language
- **PyTorch** — Deep learning framework
- **Matplotlib** — Attention weight visualization

## 📂 Project Structure

```text
Lab-4.4-Attention-Mechanism/
│
├── attention_mechanism.py
├── README.md
└── requirements.txt
```

## 📊 Dataset Description

The project generates a synthetic dataset containing 100 sequences, each with 6 randomly generated integers between 0 and 9.

```python
seqs = torch.randint(0, 10, (100, 6))
labels = (seqs.sum(1) % 2).long()
```

Each sequence receives a binary label:

- **0:** The sum of the sequence is even.
- **1:** The sum of the sequence is odd.

The dataset is generated randomly each time the program runs.

## 🧠 Model Architecture

The `AttentionClassifier` model consists of the following components:

| Layer | Configuration | Purpose |
|---|---|---|
| Embedding | 10 tokens, 16 dimensions | Converts integers into vectors |
| GRU | 16 input features, 32 hidden units | Processes sequence information |
| Attention | Linear layer, 32 → 1 | Computes a score for each position |
| Softmax | Sequence dimension | Converts scores into attention weights |
| Context Vector | Weighted sum of hidden states | Combines sequence information |
| Fully Connected | 32 → 2 | Produces scores for the two classes |

### How Attention Works

1. The embedding layer converts each integer into a vector.
2. The GRU processes the sequence and generates hidden states.
3. The attention layer assigns a score to each hidden state.
4. Softmax converts the scores into weights that sum to 1 for each sequence.
5. The weighted sum produces a context vector.
6. The final layer predicts whether the sequence sum is even or odd.

## ⚙️ Training Process

The model is trained for five epochs using the Adam optimizer and cross-entropy loss.

```python
for ep in range(5):
    out, attn = model(seqs)
    loss = F.cross_entropy(out, labels)

    opt.zero_grad()
    loss.backward()
    opt.step()

    print("epoch", ep, "loss", loss.item())
```

The loss measures the difference between the model's predictions and the correct labels. Training updates the model parameters to reduce this loss.

**Note:** Five epochs are intended for this introductory exercise. They do not guarantee high classification accuracy.

## 📈 Attention Visualization

The attention weights are visualized using Matplotlib.

```python
import matplotlib.pyplot as plt

plt.matshow(attn.detach().numpy()[:10])
plt.title("Attention Weights (10 Samples)")
plt.xlabel("Sequence Position")
plt.ylabel("Sample")
plt.show()
```

The heatmap displays attention weights for the first ten sequences. Each row represents a sequence, and each column represents one of its six positions.

Higher values indicate that the model assigns more attention weight to a particular position. These weights show the model's learned weighting, but do not necessarily explain its decisions completely.

## 🚀 Installation and Execution

### 1. Install Python

Install Python 3.10 or later. Python 3.11 is suitable for this project.

### 2. Install Dependencies

Open a terminal in your project folder and run:

```bash
python -m pip install torch matplotlib
```

Alternatively, create a `requirements.txt` file containing:

```text
torch
matplotlib
```

Then install the packages:

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Program

Save your Python code as `attention_mechanism.py` and execute:

```bash
python attention_mechanism.py
```

The program prints the dataset shapes, sample sequences, labels, model architecture, and training loss for each epoch. It then displays an attention heatmap.

## 📌 Expected Output

The dataset shapes should be:

```text
Sequences shape: torch.Size([100, 6])
Labels shape: torch.Size([100])
```

The model should print a summary of its layers and an attention model confirmation message.

Training output will contain five epoch-loss values, for example:

```text
epoch 0 loss ...
epoch 1 loss ...
epoch 2 loss ...
epoch 3 loss ...
epoch 4 loss ...
```

The exact loss values vary because the dataset and model parameters are initialized randomly.

## 🔍 Key Learnings

- Creating synthetic sequence datasets with PyTorch.
- Building a neural network using `nn.Module`.
- Processing sequential data with GRUs.
- Calculating attention scores using a linear layer.
- Applying softmax to obtain attention weights.
- Training a classifier using backpropagation.
- Visualizing attention distributions with a heatmap.

## 🔮 Future Improvements

- Increase the number of training epochs.
- Generate a larger dataset.
- Evaluate accuracy on a separate test dataset.
- Add reproducible random seeds.
- Compare the model against a baseline classifier.
- Experiment with different attention mechanisms.
- Plot the training loss over time.

## 👨‍💻 Author

**Muhammad Suffiyan Rafi**

Computer Science Graduate | AI/ML Learner

## 📄 License

This project is intended for educational and learning purposes. You may modify and extend it for practice and portfolio development.
