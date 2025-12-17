### **Task Title**
*Task 01 — Prediction Analysis*

### **1. Objective**
To evaluate the model's inference capabilities by running predictions on three specific test samples and analyzing the underlying mechanisms of the forward pass, activations, and optimization that lead to these results.

### **2. Code Used**
```
# Select samples from indices 10, 20, and 1003
indices = [10, 20, 1003]

for i in indices:
    sample = x_test[i].reshape(1, 28, 28)
    pred = model.predict(sample)
    print(f"Sample Index: {i}")
    print("Predicted Label:", np.argmax(pred))
    print("True Label:", y_test[i])
    print("-" * 20)

```
### **3. Results**
```
Sample 10: Predicted: 0 | True: 0

Sample 20: Predicted: 9 | True: 9

Sample 1003: Predicted: 5 | True: 5

```

### **4. Short Analysis**
The model successfully classified all three samples. The mechanics behind this include:

Forward Pass: The input image (28x28 pixel grid) was flattened and passed through the layers. Through matrix multiplication, the network transformed raw pixel intensities into abstract feature representations (like edges and curves).

Activation Functions: - ReLU: Applied in hidden layers to introduce non-linearity, allowing the network to learn the complex, non-linear boundaries between digits.

Softmax: Applied at the output layer to convert the raw logits into a probability distribution, identifying the class with the highest probability.

Optimizer Role: During training, the Adam optimizer dynamically adjusted the learning rates and updated the weights to minimize loss, resulting in a converged state capable of generalizing well to these unseen test samples.

### **5. Key Takeaway**
Correct predictions validate that the network has learned robust feature representations and optimal weights through the backpropagation of error and adaptive optimization.