import streamlit as st
import torch
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

CAMINHO_MODELO = Path(__file__).resolve().parent / "melhor_modelo_resnet.pth"

# Setando o mesmo modelo que foi utilizado para treinar os dados 
# (mesma arquitetura, porém sem os pesos visto que vamos carregar eles mais na frente)
model = models.resnet50(weights=None)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 1),
    torch.nn.Sigmoid()
)

# Carregando os pesos já treinados para o modelo
model.load_state_dict(torch.load(CAMINHO_MODELO, map_location="cpu"))
model.eval()

# Definindo a classe do Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, class_idx):
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = heatmap / np.max(heatmap)
        return heatmap

# Pré-processamento
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_and_explain(image, model):
    input_tensor = preprocess(image).unsqueeze(0)

    # Forward pass to get prediction
    with torch.no_grad():
        output = model(input_tensor)
        is_dog = output.item() >= 0.5
        prediction = "Cachorro" if is_dog else "Gato"
    
    # Grad-CAM setup
    grad_cam = GradCAM(model, model.layer4[2].conv3)
    output = model(input_tensor)

    # Backward pass for Grad-CAM
    model.zero_grad()
    if is_dog:
        # Backpropagate normally for "Dog" class
        output.backward(retain_graph=True)
    else:
        # Reverse the output for "Cat" class to focus on areas leading to "not Dog"
        (1 - output).backward(retain_graph=True)

    # Grad-CAM heatmap
    heatmap = grad_cam.generate_heatmap(0)  
    
    # Overlay heatmap on the image
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed_img = np.array(image.resize((224, 224))) * 0.5 + heatmap * 0.5
    overlayed_img = overlayed_img / np.max(overlayed_img)

    return prediction, overlayed_img

# Streamlit app
st.title("Gato ou cachorro?")

# Carregar a imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type="jpg")

if uploaded_file is not None:
    # Carregando a imagem
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagem carregada", use_column_width=True)

    # Prediction and Grad-CAM heatmap generation
    with st.spinner("Classificando e gerando o Grad-CAM..."):
        prediction, overlayed_img = predict_and_explain(image, model)
        
    # Mostrando os resultados
    st.write(f"Classificação: **{prediction}**")
    
    # Convert the overlayed image back to display format
    overlayed_img_pil = Image.fromarray((overlayed_img * 255).astype(np.uint8))
    st.image(overlayed_img_pil, caption="Grad-CAM Heatmap", use_column_width=True)
