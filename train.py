import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import ConvNet

def train_model(epochs=10, batch_size=64, lr=0.001):
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Entraînement sur : {device}")

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    model = ConvNet().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        print(f"Époque [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} - Accuracy: {train_acc:.2f}%")

        # Sauvegarde du meilleur modèle
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print("Entraînement terminé ! Modèle sauvegardé sous 'best_model.pth'")

if __name__ == "__main__":
    train_model(epochs=10)