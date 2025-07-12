import os
import joblib
import pandas as pd


def load_model_for_neighborhood(neighborhood: str):
    """
    Carga el modelo correspondiente al barrio.

    Args:
        neighborhood (str): Nombre del barrio (ej. 'chamberi')

    Returns:
        model: Modelo de predicción
    """
    normalized_neighborhood = neighborhood.strip().replace(" ", "_").lower()
    model_path = f"modelos/modelo_{normalized_neighborhood}.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo para el barrio: {neighborhood} en la ruta {model_path}")

    model = joblib.load(model_path)
    return model


def predecir_precio(df_preprocesado: pd.DataFrame, neighborhood: str) -> float:
    """
    Predice el precio de la propiedad usando el modelo del barrio.

    Args:
        df_preprocesado (pd.DataFrame): DataFrame con una sola fila ya preprocesada
        neighborhood (str): Nombre del barrio

    Returns:
        float: Precio predicho
    """
    if df_preprocesado.shape[0] != 1:
        raise ValueError("El DataFrame preprocesado debe contener exactamente una fila.")

    modelo = load_model_for_neighborhood(neighborhood)
    prediccion = modelo.predict(df_preprocesado)[0]
    return float(prediccion)

