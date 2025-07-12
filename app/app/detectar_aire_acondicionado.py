import streamlit as st
import os
from roboflow import Roboflow
from inference_sdk import InferenceHTTPClient
def detectar_aire_acondicionado_en_imagenes(image_list, st=None):
    """
    Detecta aire acondicionado en una lista de imágenes.
    Devuelve 1 si detecta aire acondicionado en al menos 2 imágenes, 0 en caso contrario.

    Parámetros:
    - image_list: lista de imágenes, cada una puede ser ruta (str) o bytes
    - st: streamlit para logs (opcional)

    Retorna:
    - int: 1 o 0
    """

    def log(msg):
        if st:
            st.write(msg)
        else:
            print(msg)

    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="EYfHRlA9fPiyTkjW6pKs"
    )

    detecciones_positivas = 0

    if not image_list:
        log(" No se recibieron imágenes para detectar aire acondicionado.")
        return 0

    for i, img in enumerate(image_list):
        try:
            # img puede ser bytes o path (string)
            if isinstance(img, bytes):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".webp") as tmp:
                    tmp.write(img)
                    tmp.flush()
                    resultado = CLIENT.infer(tmp.name, model_id="test3-qzutu/3")
            else:
                resultado = CLIENT.infer(img, model_id="test3-qzutu/3")

            if resultado and resultado.get("predictions"):
                detectado = False
                for prediction in resultado["predictions"]:
                    if prediction.get("class") == 'Air conditioning 1':
                        detecciones_positivas += 1
                        detectado = True
                        log(f"✅ Aire acondicionado detectado en imagen {i}")
                        break
                if not detectado:
                    log(f"❌ No se detectó aire acondicionado en imagen {i}")
            else:
                log(f"❌ No se detectaron predicciones en imagen {i}")

            if detecciones_positivas >= 2:
                log(f" Aire acondicionado confirmado en al menos 2 imágenes. Parando detección.")
                break

        except Exception as e:
            log(f"⚠️ Error procesando imagen {i}: {str(e)}")

    if detecciones_positivas < 2:
        log(f" Aire acondicionado detectado en {detecciones_positivas} imágenes, no alcanza el umbral de 2.")

    return 1 if detecciones_positivas >= 2 else 0

