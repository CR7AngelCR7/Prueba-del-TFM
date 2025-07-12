import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from collections import Counter
import os
import streamlit as st
import io
from altura_techo_estimator import estimate_ceiling_height
from reformado_classifier import estimate_reformado_from_images, load_reformado_model
from detectar_aire_acondicionado import detectar_aire_acondicionado_en_imagenes
    
def add_features(data_dict, neighborhood, image_list=None):
    import pandas as pd

    df = pd.DataFrame([data_dict])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if neighborhood in ["chamberi", "arganzuela"]:
    	model = load_reformado_model(device=device)
    	reformado = estimate_reformado_from_images(image_list or [], model, device=device)
    	df["reformado_bin"] = reformado


    if neighborhood == "retiro":
    	altura_estim, logs = estimate_ceiling_height(image_list)
    	for msg in logs:
    		st.write(msg)
    	st.write(f"Altura estimada: {altura_estim}")
    	df["altura_techo"] = altura_estim
    	
    if neighborhood == "centro":
    	aire = detectar_aire_acondicionado_en_imagenes(image_list, st=st)
    	df["aire_acondicionado_bin"] = aire


    return df
