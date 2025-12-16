import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.cnn.mobilenet_emotion import MobileNetEmotion
from src.preprocessing.dataset import get_dataloaders


def train(model, train_loader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

        torch.save(
            model.state_dict(),
            "checkpoints/mobilenet_emotion.pth"
        )
        print("💾 Checkpoint saved")


def main():
    print("🚀 Training started")

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("🖥 Using device:", device)

    train_loader, _ = get_dataloaders(
        data_dir="data/fer2013",
        batch_size=16
    )

    print("📦 Dataset loaded")
    print("🔢 Batches per epoch:", len(train_loader))

    model = MobileNetEmotion(num_classes=7)
    print("🧠 Model initialized")

    train(model, train_loader, device)

    print("🎉 Training finished successfully")


if __name__ == "__main__":
    main()