import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from model import ConvNet

# Détection automatique du device (MPS pour ton Mac M1 Pro, sinon CUDA ou CPU)
device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

classes = ['avion', 'automobile', 'oiseau', 'chat', 'cerf', 'chien', 'grenouille', 'cheval', 'bateau', 'camion']

# Chargement du modèle
model = ConvNet().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Transformations d'images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict(image):
    if image is None:
        return None
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    return {classes[i]: float(probabilities[i]) for i in range(10)}

# Interface personnalisée avec Blocks (Évite le bug Gradio/FastAPI)
with gr.Blocks(title="Démo Classificateur CIFAR-10") as demo:
    gr.Markdown("# 🚀 Démo Classificateur PyTorch CIFAR-10")
    gr.Markdown("Glisse une image ci-dessous pour obtenir les prédictions du réseau de neurones en temps réel.")
    
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Image d'entrée")
            btn = gr.Button("Classifier", variant="primary")
        with gr.Column():
            label_output = gr.Label(num_top_classes=3, label="Résultats")
            
    btn.click(fn=predict, inputs=img_input, outputs=label_output)

if __name__ == "__main__":
    demo.launch()