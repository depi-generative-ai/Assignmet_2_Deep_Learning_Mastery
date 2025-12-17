### **Task Title**
*Task 10 Weights*

### **1. Objective**
To peek inside the "black box" by extracting the learnable parameters (weights) of the first Dense layer and analyzing how different regularization techniques (EarlyStopping, Dropout, L2) fundamentally alter the magnitude and distribution of these weights.

### **2. Code Used**
```
# Extract weights (w) and biases (b) from the first hidden layer
w, b = model.layers[1].get_weights()
print(f"Weight Shape: {w.shape}")

# Convert to DataFrame for statistical summary
df_w = pd.DataFrame(w)
print(df_w.describe().T)
```

### **3. Results**
| Model Configuration | Mean Weight | Std Dev | Max Weight | Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Standard (No Reg)** | ~ -0.01 | ~ 0.22 | **0.87** | **High Variance:** Weights grew freely to fit training data. |
| **EarlyStopping (Pat=3)** | ~ -0.02 | ~ 0.14 | **0.56** | **Restricted:** Training halted before weights could grow large to fit noise. |
| **Dropout (30%)** | ~ -0.01 | ~ 0.16 | **0.66** | **Distributed:** Weights are moderate; redundancy is encouraged. |
| **L2 (0.01)** | ~ 0.0002 | **0.01** | **0.09** | **Suppressed:** Weights are near zero due to the heavy penalty. |

### **4. Short Analysis**
Why so many parameters? The input is a $28 \times 28$ image flattened to 784 pixels. Each of the 128 neurons must learn a unique weight for every single pixel, resulting in $784 \times 128 \approx 100k$ parameters. This massive capacity allows the model to memorize the training set easily if uncontrolled.

Effect of L2 Regularization:  The difference is stark. In the L2 model, the standard deviation dropped to 0.01 (vs 0.22 in standard). The optimizer was forced to keep weights tiny to minimize the loss function's penalty term ($\lambda \sum w^2$). This prevents any single feature from dominating the decision.

Effect of EarlyStopping: By stopping at Epoch 12 (instead of 20), the weights didn't have time to drift further away from initialization to fit the finer noise details, resulting in smaller weights (Max 0.56 vs 0.87).

Effect of Dropout: Dropout creates a "smearing" effect. Since neurons are randomly killed, the remaining weights must compensate, preventing the model from relying on a single large weight connection (co-adaptation).

### **5. Key Takeaway**
Overfitting manifests physically as large, exploded weights; L2 regularization explicitly suppresses this by penalty, while EarlyStopping implicitly prevents it by limiting the learning time.