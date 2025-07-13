import pandas as pd
from pycaret.regression import load_model, predict_model, setup, finalize_model

# Cache para no cargar y preparar setup cada vez
_model_cache = {}

def load_and_finalize_model(neighborhood: str):
    global _model_cache

    if neighborhood in _model_cache:
        return _model_cache[neighborhood]

    # Carga dataset completo para recrear el setup
    df = pd.read_csv('modelos/datos_imagenes_con_predicciones_y_aire.csv')

    energy_certificate_order = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    energy_certificate_mapping = {cert: i for i, cert in enumerate(energy_certificate_order)}
    df['energy_certificate_encoded'] = df['energy_certificate'].str.lower().map(energy_certificate_mapping)

    features = [
        'bathroom_count', 'bedroom_count', 'floor',
        'latitude', 'longitude', 'lot_size',
        'property_type', 'neighborhood', 'exterior', 'ascensor',
        'reformado_bin', 'energy_certificate_encoded'
    ]

    df_training = df[features + ['property_price']].copy()

    # Categóricas iguales que en el entrenamiento
    df_training['property_type'] = df_training['property_type'].astype('category')
    df_training['neighborhood'] = df_training['neighborhood'].astype('category')

    # Setup con parámetros iguales que usaste en jupyter
    s = setup(data=df_training, target='property_price', session_id=123, verbose=False)

    # Carga modelo
    normalized_neighborhood = neighborhood.strip().replace(" ", "_").lower()
    model_path = f"modelos/modelo_{normalized_neighborhood}"

    modelo = load_model(model_path)

    # Finaliza el modelo para evitar errores al predecir
    modelo_finalizado = finalize_model(modelo)

    _model_cache[neighborhood] = modelo_finalizado

    return modelo_finalizado

def predecir_precio(df_preprocesado: pd.DataFrame, neighborhood: str) -> float:
    if df_preprocesado.shape[0] != 1:
        raise ValueError("El DataFrame debe contener exactamente una fila")

    modelo = load_and_finalize_model(neighborhood)

    # En tu preprocesamiento debes asegurarte que df_preprocesado tiene:
    # 'property_type' y 'neighborhood' como categorías
    df_preprocesado['property_type'] = df_preprocesado['property_type'].astype('category')
    df_preprocesado['neighborhood'] = df_preprocesado['neighborhood'].astype('category')

    # Predict con el DataFrame tal cual, sin filtrar columnas
    pred = predict_model(modelo, data=df_preprocesado)

    return float(pred['prediction_label'].iloc[0])

