import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import io  # ← IMPORTANTE
import streamlit as st  # si estás usando Streamlit

# -----------  Cargar modelo EfficientNet entrenado con torchvision -----------
def load_reformado_model(path="modelos/efficientnet_reformado.pth", device="cpu"):
    st.write(" Cargando modelo de reformado...")
    
    # Cargar arquitectura desde torchvision
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # salida 2 clases

    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    st.write(" Modelo de reformado cargado.")
    return model

# -----------  Realizar predicciones sobre imágenes -----------
def estimate_reformado_from_images(image_list, model, device="cpu"):
    st.write(f" Número de imágenes recibidas: {len(image_list)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Usa la normalización usada en el entrenamiento
    ])

    preds = []

    for i, img_bytes in enumerate(image_list):
        st.write(f" Procesando imagen {i}")
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except Exception as e:
            st.write(f"⚠️ Imagen {i} no válida: {e}")
            continue

        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)[0, 1].item()  # probabilidad de clase 1
            pred_class = int(prob > 0.5)
            st.write(f"✅ Imagen {i} → prob: {prob:.2f} → clase: {pred_class}")
            preds.append(pred_class)

    if not preds:
        st.write("⚠️ No se pudieron predecir clases. Valor por defecto: 0 (no reformado)")
        return 0

    st.write(f" Predicciones: {preds}")
    majority = int(np.round(np.mean(preds)))  # mayoría simple
    st.write(f" Clase mayoritaria final: {majority}")
    return majority

