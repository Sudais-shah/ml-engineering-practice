import numpy as np 

# Generate data
np.random.seed(42)
n_samples = 100
n_features = 1
X = np.random.rand(n_samples, n_features)
true_w = np.array([3.0])
true_b = 2.0
y = X @ true_w + true_b + np.random.randn(n_samples) * 0.1

# initialize parameters
w = np.zeros(X.shape[1])
b = 0.0

# training loop 
epochs = 100 
learning_rate = 0.1
lambda_ = 0.1

for epoch in range(epochs):
    y_hat = X @ w + b 
    loss = np.mean((y - y_hat) **2) + (lambda_ / (2 * len(y))) * np.sum(w ** 2)

    dw = (2 / len(y)) * (X.T @ (y_hat - y)) + (lambda_ / len(y)) * w 
    db = (2 / len(y)) * np.sum(y_hat - y)

    w = w - learning_rate * dw 
    b = b - learning_rate * db

    print(f"epoch {epoch}, loss {loss}")
