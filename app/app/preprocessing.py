import pandas as pd
import os
import joblib
def preprocess_input(df: pd.DataFrame, neighborhood: str) -> pd.DataFrame:
    df = df.copy()

    # Eliminar columnas que no se usan en el modelo
    for col_to_drop in ['address', 'agency_name', 'images','neighborhood','property_description', 'property_id','property_title']:
        if col_to_drop in df.columns:
            df.drop(columns=[col_to_drop], inplace=True)

    # Codificar energy_certificate si existe
    energy_certificate_order = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    energy_certificate_mapping = {cert: i for i, cert in enumerate(energy_certificate_order)}

    if 'energy_certificate' in df.columns:
        df['energy_certificate_encoded'] = df['energy_certificate'].str.lower().map(energy_certificate_mapping)
        df.drop(columns=['energy_certificate'], inplace=True)
    else:
        df['energy_certificate_encoded'] = 0

    # Dummy simple para property_type_flat
    if 'property_type' in df.columns:
        df['property_type_flat'] = (df['property_type'].str.lower() == 'flat').astype(int)
        df.drop(columns=['property_type'], inplace=True)

    # Rellenar NaNs numéricos con 0
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Rellenar categóricos con 'desconocido'
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('desconocido')

    # Añadir la columna del barrio correspondiente a 1
    if neighborhood == 'chamberi':
        df['neighborhood_Chamberí'] = 1
        # columnas esperadas para Chamberí
        model_features = ['bathroom_count', 'bedroom_count', 'floor', 'latitude', 'longitude',
                          'lot_size', 'property_type_flat', 'neighborhood_Chamberí', 'exterior',
                          'ascensor', 'energy_certificate_encoded', 'reformado_bin']
    elif neighborhood == 'centro':
        df['neighborhood_Centro'] = 1
        model_features = ['bathroom_count', 'bedroom_count', 'floor', 'latitude', 'longitude',
                          'lot_size', 'property_type_flat', 'neighborhood_Centro', 'exterior',
                          'ascensor', 'energy_certificate_encoded', 'aire_acondicionado']
    elif neighborhood == 'arganzuela':
        df['neighborhood_Arganzuela'] = 1
        model_features = ['bathroom_count', 'bedroom_count', 'floor', 'latitude', 'longitude',
                          'lot_size', 'property_type_flat', 'neighborhood_Arganzuela', 'exterior',
                          'ascensor', 'energy_certificate_encoded', 'reformado_bin']
    elif neighborhood == 'retiro':
        df['neighborhood_Retiro'] = 1
        model_features = ['bathroom_count', 'bedroom_count', 'floor', 'latitude', 'longitude',
                          'lot_size', 'property_type_flat', 'neighborhood_Retiro', 'exterior',
                          'ascensor', 'energy_certificate_encoded', 'altura_techo']
    else:
        # Barrio desconocido: solo columnas base + dummy de barrio (todo cero)
        model_features = ['bathroom_count', 'bedroom_count', 'floor', 'latitude', 'longitude',
                          'lot_size', 'property_type_flat', 'exterior', 'ascensor', 'energy_certificate_encoded']
                       

    # Asegurar que todas las columnas están presentes, si no crearlas con 0
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    # Reordenar columnas según modelo para el barrio
    df = df[model_features]
    print("Columnas en df antes de retornar:", df.columns.tolist())


    return df

