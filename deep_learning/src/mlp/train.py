import torch
import torch.nn as nn
import MLP
# -------------------------------
# FAKE DATA (SIMULATING REAL DATA)
# -------------------------------
batch_size = 32
input_dim = 784      # number of features per sample
hidden_dim = 64      # number of neurons in hidden layer
output_dim = 10      # number of classes

# Input data (32 samples, each with 784 features)
x = torch.randn(batch_size, input_dim)

# Target labels (correct answers for each sample)
y = torch.randint(0, output_dim, (batch_size,))


# -------------------------------
# INITIALIZE MODEL
# -------------------------------
model = MLP(input_dim, hidden_dim, output_dim)

print(model)


# -------------------------------
# LOSS FUNCTION & OPTIMIZER
# -------------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)


# -------------------------------
# TRAINING LOOP
# -------------------------------
epochs = 5

for epoch in range(epochs):

    # 1. Forward pass (model makes predictions)
    outputs = model(x)

    # 2. Calculate loss (how wrong the model is)
    loss = loss_fn(outputs, y)

    # 3. Clear previous gradients
    optimizer.zero_grad()

    # 4. Backward pass (compute gradients)
    loss.backward()

    # 5. Update weights
    optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")
