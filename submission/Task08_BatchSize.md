### **Task Title**
*Task 08 BatchSize*

### **1. Objective**
To investigate how batch size (8, 32, 128) influences the stochastic nature of gradient descent, analyzing the trade-off between "gradient noise" (which helps escape local minima) and training stability/speed.

### **2. Code Used**
```
# Configuration: batches=8 the same concept for the rest of the batches 
model_with_8_batches = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.1),   # prevent overfitting
    keras.layers.Dense(10, activation="softmax")
])

model_with_8_batches.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
]

history = model_with_8_batches.fit(
    x_tr, y_tr,
    epochs=20,
    batch_size=8,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)
```

### **3. Results**
| Batch Size | Steps per Epoch | Time per Epoch | Best Val Loss | Observation |
| :--- | :--- | :--- | :--- | :--- |
| **8** | 6,875 | ~17s | 0.0837 | **Noisy:** High variance updates; converged fast in epochs but slow in wall-clock time. |
| **32** | 1,719 | ~5s | 0.0622 | **Balanced:** Best generalization; optimal trade-off between noise and stability. |
| **128** | 430 | ~1s | 0.0657 | **Smooth:** Fastest computation; smoother convergence but required more epochs to settle. |

### **4. Short Analysis**
The batch size fundamentally alters the "path" the optimizer takes through the loss landscape:

Gradient Noise & Small Batches (8): With only 8 samples, the gradient estimate is noisy (high variance). While this noise can help the model "jump" out of sharp local minima, here it was slightly excessive, leading to a "jittery" path that prevented the model from settling into the absolute deepest valley of the loss function (Val Loss 0.08 vs 0.06).

Computational Speed vs. Convergence (128): Larger batches utilize GPU vectorization better (1s vs 17s per epoch). However, because there are fewer weight updates per epoch (430 vs 6875), the model "learns" less per epoch, requiring more total epochs to reach similar accuracy.

Generalization Sweet Spot (32): The batch size of 32 provided enough noise to prevent overfitting to sharp valleys but enough stability to converge efficiently. It achieved the lowest validation loss (0.0622), suggesting it found a flatter, more robust minimum.

### **5. Key Takeaway**
Small batches update weights frequently with high variance (good for exploration, bad for speed), while large batches update infrequently with low variance (good for speed, bad for exploration); moderate batch sizes (32-64) usually offer the best generalization.