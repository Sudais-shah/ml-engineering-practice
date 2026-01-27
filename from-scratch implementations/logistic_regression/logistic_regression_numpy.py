import numpy as np 
# -----------------------------
# Generate synthetic binary data
# -----------------------------
np.random.seed(42)
n_samples = 100
n_features = 2

# Features
X = np.random.rand(n_samples, n_features)

# True weights and bias (for generating labels)
true_w = np.array([2.0, -3.0])
true_b = 0.5

# Linear combination + sigmoid to generate probabilities
linear_comb = X @ true_w + true_b
prob = 1 / (1 + np.exp(-linear_comb))

# Convert probabilities to binary labels
y = (prob > 0.5).astype(int)

# -----------------------------
# Initialize parameters
# -----------------------------
w = np.zeros(n_features)
b = 0.0

# -----------------------------
# Hyperparameters
# -----------------------------
epochs = 1000
learning_rate = 0.1
lambda_ = 0.01
patience = 10  # for early stopping
best_loss = float('inf')
counter = 0

# BCE Loss
## -1 * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
m = len(y)

for epoch in range(epochs):
    y_hat = 1 / ( 1 + np.exp( - (X @ w + b )))
    loss = - np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) + lambda_ / (2 * m) * np.sum(w **2)

    dw = (1 / m) * X.T @ (y_hat - y) + lambda_ / m * w
    db = (1 / m) * np.sum(y_hat - y)

    w -= learning_rate * dw
    b -= learning_rate * db

    if loss < best_loss:
        best_loss = loss 
        w_best = w.copy()
        b_best = b
        counter = 0
    else:
        counter += 1

    if counter >= patience:
            print(f"Early stopping at epoch {epoch}.")

    if epoch % 100 == 0 or epoch == epochs - 1:
         print(f"epoch {epoch}, loss {loss:.4f}")
        