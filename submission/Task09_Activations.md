### **Task Title**
*Task 09 Activations*

### **1. Objective**
To evaluate how different activation functions (Tanh, Softsign, GELU) impact gradient flow and convergence speed compared to the standard ReLU, and to understand why modern architectures (like Transformers) prefer GELU.

### **2. Code Used**
```
# Configuration: activation function="tanh" the same concept for the rest of the activation functions 

model_with_Tanh = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="tanh"),
    keras.layers.Dropout(0.1),   # prevent overfitting
    keras.layers.Dense(10, activation="softmax")
])

model_with_Tanh.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
]

history = model_with_Tanh.fit(
    x_tr, y_tr,
    epochs=20,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)
```

### **3. Results**
| Activation | Best Val Loss | Best Val Acc | Converged At | Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Tanh** | 0.0695 | 98.08% | Epoch 11 | **Stable:** Good performance but slightly slower convergence than GELU. |
| **Softsign** | 0.0725 | 98.02% | Epoch 17 | **Slow:** Took significantly more epochs to reach the same accuracy levels. |
| **GELU** | 0.0664 | 98.38% | Epoch 9 | **Superior:** Fastest convergence and highest accuracy. |

### **4. Short Analysis**
The choice of activation function fundamentally changes how gradients propagate through the network:

GELU (Gaussian Error Linear Unit): The standout performer. Unlike ReLU which hard-clips negative values to zero, GELU provides a smooth, probabilistic curve that allows small negative values to propagate. This smoothness makes the loss landscape easier to traverse for the optimizer, explaining its dominance in Transformers (BERT/GPT) and why it achieved the lowest loss (0.0664) here.

Tanh (Hyperbolic Tangent): Tanh is zero-centered (outputs range -1 to 1), which helps center the data for the next layer. However, it suffers from the vanishing gradient problem at the extremes (where the curve flattens out), which limits how deep networks can go compared to non-saturating functions like ReLU.

Softsign: A smoother alternative to Tanh that approaches its asymptotes polynomially rather than exponentially. While theoretically robust, it proved computationally inefficient here, requiring nearly double the epochs (17) to reach convergence compared to GELU.

Why ReLU Remains Popular: Despite GELU's superior performance, ReLU remains the default for CNNs and MLPs because it is computationally essentially free ( max(0, x) requires no expensive exponential calculations).

### **5. Key Takeaway**
While GELU provides the best theoretical and empirical performance for complex models by preserving gradient flow, ReLU remains the efficiency king for standard tasks where computational cost is a priority.