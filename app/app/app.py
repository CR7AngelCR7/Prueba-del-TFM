import os
import joblib
import traceback
from input_handler import get_user_inputs
from feature_engineering import add_features
from preprocessing import preprocess_input
from predictor import predecir_precio, load_model_for_neighborhood
import io
import streamlit as st


st.set_page_config(page_title="Predicción de precios", page_icon="🏡")
st.title("Predicción del precio de una propiedad 🏡")

# --- 1. Entrada del usuario ---
user_input = get_user_inputs()

if user_input:
    try:
        st.success(" Datos del usuario recibidos")
        neighborhood = user_input["neighborhood"]
        images = user_input.get("images", [])

        # Convertir UploadedFile a bytes para el modelo
        image_bytes_list = [img.getvalue() for img in images] if images else []

        st.write(f" Barrio seleccionado: {neighborhood}")
        st.write(f" Nº de imágenes cargadas: {len(image_bytes_list)}")

        # Crear copia sin imágenes para pasar a add_features
        user_input_clean = user_input.copy()
        user_input_clean.pop("images", None)

        data_with_features = add_features(user_input_clean, neighborhood, image_list=image_bytes_list)

        # Mostrar sin columna images (que no existe ya)
        st.write(data_with_features)
 # --- 3. Preprocesamiento general ---
        st.write(" Preprocesando datos...")
        processed_data = preprocess_input(data_with_features, neighborhood)
        st.success(" Preprocesamiento completado")
        st.write(processed_data)
        
        st.write(" Realizando predicción...")
        precio_estimado = predecir_precio(processed_data, neighborhood)
        st.success(f" Precio estimado: {precio_estimado:,.0f} €")

 

    except Exception as e:
        st.error("❌ Ocurrió un error en la ejecución:")
        st.code(traceback.format_exc())
