import pandas as pd

def preprocess_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesa una fila de entrada para predicción con un modelo de PyCaret.
    NO realiza transformaciones avanzadas (OneHot, escalado...), solo lo mínimo necesario.

    Args:
        df_raw (pd.DataFrame): Una fila de datos crudos

    Returns:
        pd.DataFrame: Fila lista para `predict_model()`
    """
    # Mapear certificado energético si es necesario
    energy_certificate_order = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    energy_certificate_mapping = {cert: i for i, cert in enumerate(energy_certificate_order)}

    df = df_raw.copy()

    # Asegurar que la columna existe y mapear si hace falta
    if 'energy_certificate' in df.columns:
        df['energy_certificate_encoded'] = df['energy_certificate'].str.lower().map(energy_certificate_mapping)

    # Definir las features usadas para entrenar
    features = [
        'bathroom_count', 'bedroom_count', 'floor',
        'latitude', 'longitude', 'lot_size',
        'property_type', 'neighborhood', 'exterior', 'ascensor',
        'reformado_bin', 'energy_certificate_encoded'
    ]

    # Añadir columnas faltantes como NaN o 0 por defecto
    for col in features:
        if col not in df.columns:
            if col == 'reformado_bin':
                df[col] = 0  # valor por defecto para variable binaria
            else:
                df[col] = pd.NA  # o np.nan

    # Reordenar las columnas según el orden original
    df = df[features]

    return df

