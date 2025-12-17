# 🧠 Deep Learning Dynamics: An Ablation Study on MNIST

![Banner Image](https://img.shields.io/badge/TensorFlow-2.x-orange.svg) ![Banner Image](https://img.shields.io/badge/Python-3.8+-blue.svg) ![Banner Image](https://img.shields.io/badge/Status-Complete-green.svg)

## **1. Project Overview**
This project serves as a comprehensive investigation into the internal mechanics of Artificial Neural Networks (ANNs). Using the **MNIST handwritten digit dataset** as a baseline, we conduct **10 controlled experiments (Ablation Studies)** to isolate and analyze the effects of specific hyperparameters, regularization techniques, and architectural choices.

The primary objective is not merely to maximize accuracy, but to visualize and understand the **training dynamics**—specifically how models converge, why they overfit, and how different optimizers navigate the loss landscape.

---

## **2. Repository Structure**
The project is organized to separate code, raw results, and analytical reports.

```text
deep-learning-dynamics/
├── notebook.ipynb             # The primary Jupyter Notebook containing all 10 experiments
├── README.md                  # Project documentation (this file)
├── image.png/                     # Raw custom handwriting samples for testing
├── image_2.png
├── image_3.png                    
│   
├── results/                   # Generated visualizations and plots
│   ├── predictions/           # Inference outputs and heatmaps
│   ├── loss_curves/           # Training vs. Validation loss plots
│   └── comparisons/           # Bar charts comparing optimizer performance
└── submission/                # Detailed Markdown analysis files for each task
    ├── Task01_PredictionAnalysis.md   # Forward pass & Activation analysis
    ├── Task02_CustomDigit.md          # Out-of-distribution generalization
    ├── Task03_Epochs.md               # Overfitting detection
    ├── Task04_EarlyStopping.md        # Regularization via stopping criteria
    ├── Task05_Dropout.md              # Neuron co-adaptation study
    ├── Task06_L2.md                   # Weight decay analysis
    ├── Task07_Optimizers.md           # Adam vs. SGD vs. Momentum
    ├── Task08_BatchSize.md            # Gradient noise trade-offs
    ├── Task09_Activations.md          # ReLU vs. GELU vs. Tanh
    └── Task10_Weights.md              # Weight matrix capacity analysis
```

## **3. Methodology & Experiment Details**
The core architecture used across most experiments is a fully connected Multi-Layer Perceptron (MLP):

*   **Input Layer:** 784 neurons (Flattened 28x28 image)
*   **Hidden Layer:** 128 neurons (Variable Activation: ReLU/GELU/Tanh)
*   **Output Layer:** 10 neurons (Softmax Activation)
*   **Loss Function:** Sparse Categorical Crossentropy

### **Phase I: Inference & Generalization**
We assessed the model's ability to interpret data outside of its training set.
*   **Task 01:** Analyzed the raw probability distributions (logits) output by the Softmax layer.
*   **Task 02:** Created custom handwritten digits to test distribution shift. We found that the model is highly sensitive to centering and padding, failing when digits were not preprocessed exactly like the MNIST training set.

### **Phase II: Regularization & Stability**
We investigated methods to prevent the model from memorizing noise (overfitting).
*   **Task 03 (Epochs):** Identified the "Generalization Gap." Training loss consistently decreased, but validation loss began diverging (increasing) after Epoch 6.
*   **Task 04 (EarlyStopping):** Implemented `patience=3` to automatically halt training when validation loss stagnated, effectively saving computational resources.
*   **Task 05 (Dropout):** Comparison of 0%, 10%, and 30% dropout rates.
    *   *Result:* 30% dropout yielded the most robust model by forcing the network to learn redundant feature representations.
*   **Task 06 (L2 Regularization):** Applied Ridge Regression ($\lambda=0.01$). This successfully suppressed weight magnitudes, forcing the model to learn smoother decision boundaries.

### **Phase III: Hyperparameter Tuning**
We benchmarked the critical "knobs" of the training process.
*   **Task 07 (Optimizers):**
    *   **SGD:** Slow, steady convergence.
    *   **Adam:** Extremely fast initial learning but prone to early plateauing.
    *   **AdamW:** The best performer, combining Adam's speed with decoupled weight decay.
*   **Task 08 (Batch Size):**
    *   **Small (8):** Noisy gradients helped escape local minima but slowed down training.
    *   **Large (128):** Fast but converged to a sharper, less generalizable solution.
    *   **Optimal (32):** Best balance of noise and stability.
*   **Task 09 (Activations):** Replaced ReLU with GELU (Gaussian Error Linear Unit). GELU achieved faster convergence due to its smooth, non-monotonic nature, which avoids the "Dead ReLU" problem.

## **4. Key Results & Visualizations**
*   **Impact of Batch Size on Gradient Noise:** Smaller batches create "jittery" loss curves, while large batches smooth out the learning process.
*   **The Effect of Dropout:** Visualizing how random neuron deactivation narrows the gap between training and validation accuracy.
*   **Weight Distribution Analysis:** Histograms showing how L2 Regularization forces weights closer to zero compared to an unregularized model.

## **5. How to Run**
Clone the repository:
```bash
git clone https://github.com/depi-generative-ai/Assignmet_2_Deep_Learning_Mastery
cd deep-learning-dynamics
```

Create a Virtual Environment (Optional but Recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install Dependencies:
```bash
pip install tensorflow
pip install pandas
pip install numpy
pip install matplotlib
pip install opencv-python
```

Launch Jupyter Notebook:
```bash
jupyter notebook notebook.ipynb
```

---
**Author:** [Ahmed Ahmed Mokhtar]

**Course:** Genrative AI