import torch
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import ConvNet
from sklearn.metrics import classification_report, confusion_matrix

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_dataloaders()

    model = ConvNet().to(device)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n--- Rapport de Classification ---")
    print(classification_report(all_labels, all_preds))
    
    print("\n--- Matrice de Confusion ---")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # Génération et sauvegarde du graphique
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')

    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    evaluate()