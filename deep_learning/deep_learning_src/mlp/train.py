import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from models import MLP
import os 

# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        outputs = model(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


# ============================================================
# MAIN
# ============================================================
def main():
    # -------------------- CONFIG --------------------
    batch_size = 64
    learning_rate = 0.1
    epochs = 10
    checkpoint_dir = "checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- DATA --------------------
    transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x.view(-1))])

    full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_dataset, val_dataset = random_split(full_train, [54000, 6000])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------- MODEL --------------------
    model = MLP(input_dim=784, output_dim=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # -------------------- CHECKPOINT TRACKER --------------------
    best_val_loss = float("inf")

    # -------------------- TRAIN LOOP --------------------
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # -------- CHECKPOINT --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"epoch": epoch + 1,"model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "best_val_loss": best_val_loss},checkpoint_path )
            print(" Saved best model")

        print(f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # -------------------- TEST BEST MODEL --------------------
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()