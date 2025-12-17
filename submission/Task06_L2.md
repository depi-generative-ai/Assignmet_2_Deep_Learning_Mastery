### **Task Title**
*Task 06 L2*

### **1. Objective**
To investigate the impact of L2 regularization (Ridge) on model weights and generalization by testing three penalty strengths ($0.0001, 0.001, 0.01$) and analyzing how penalizing large weights prevents overfitting.

### **2. Code Used**
```
# Configuration: L2 = 0.0001
model_with_l2_1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu",kernel_regularizer=keras.regularizers.l2(0.0001)),
    keras.layers.Dropout(0.1),   # prevent overfitting
    keras.layers.Dense(10, activation="softmax")
])

model_with_l2_1.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
]

history = model_with_l2_1.fit(
    x_tr, y_tr,
    epochs=20,
    batch_size=32,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)
```

### **3. Results**
| L2 Strength | Final Val Loss | Final Val Acc | Observation |
| :--- | :--- | :--- | :--- |
| **0.0001** | 0.1051 | 98.24% | **Optimal:** Best balance. High accuracy, stable loss. |
| **0.001** | 0.1641 | 97.44% | **Constrained:** Loss is higher due to penalty term added to cost. |
| **0.01** | 0.2561 | 96.84% | **Underfitting:** Penalty is too strong; model struggles to learn. |

### **4. Short Analysis**
Mechanism (Weight Decay): L2 regularization adds a penalty term $\lambda \sum w^2$ to the loss function. This forces the optimizer to shrink weights towards zero during backpropagation.

Why Smaller Weights Improve Generalization: Large weights allow the model to make sharp, complex decision boundaries that fit noise (overfitting). By constraining weights to be small, the decision boundaries become smoother and simpler, which generalizes better to unseen data.

Validation Loss Trend: As the L2 penalty increased from 0.0001 to 0.01, the validation loss increased. This is expected because the "loss" now includes the regularization penalty. However, raw accuracy decreased in the 0.01 run, indicating the regularization was too aggressive, causing the model to underfit (bias increased).

### **5. Key Takeaway**
L2 regularization simplifies the model by suppressing large weights, but setting the penalty ($\lambda$) too high can prevent the model from learning essential patterns (underfitting).