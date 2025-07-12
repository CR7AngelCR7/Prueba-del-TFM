# Predicción de Precios de Propiedades en Madrid

Este proyecto desarrolla un sistema automatizado para predecir el precio de propiedades en Madrid,
combinando datos scrappeados desde Idealista, información geográfica enriquecida (como la distancia
a estaciones de metro) y análisis visual mediante técnicas de Computer Vision.

El objetivo es generar una herramienta robusta y objetiva para el sector PropTech, capaz de valorar
viviendas a partir de múltiples fuentes de información estructurada y no estructurada.

-------------------------------------------------------
Estructura de Notebooks
-------------------------------------------------------

1. get_dataset.ipynb
---------------------
- Une datasets descargados por barrios.
- Limpieza general e imputación de valores faltantes.
- Carga y visualización básica de imágenes asociadas a cada propiedad.
- Archivos generados:
  - datos.csv: Dataset original + limpieza + EDA.
  - datos_procesados.csv: Versión sin outliers de precio.

2. AutoML_pruebas.ipynb
------------------------
- Comparativa de modelos con PyCaret, LazyPredict y AutoML.
- Evaluación de algoritmos y combinación de predictores.
- No genera nuevos archivos, solo pruebas internas.

3. ModeloSinFotos.ipynb
------------------------
- Análisis por barrio (EDA específico).
- Cálculo de distancias a estaciones de metro (vía OSMnx).
- Modelado usando únicamente variables estructurales (sin imágenes).

4. ModeloConFotos.ipynb
------------------------
- Estimación de altura de techos mediante imágenes.
- Clasificación automática del estado de reforma (modelo CNN).
- Incorporación de variables visuales como tipo de suelo, aire acondicionado, etc.
- Modelado completo con features visuales + estructurales.
- Archivo generado:
  - datos_imagenes_con_predicciones_y_aire.csv

-------------------------------------------------------
CSVs disponibles
-------------------------------------------------------

| Archivo                                     | Descripción                                         |
|---------------------------------------------|-----------------------------------------------------|
| datos.csv                                   | Datos scrappeados + EDA inicial                     |
| datos_procesados.csv                        | Datos sin outliers en precio                        |
| datos_imagenes_con_predicciones_y_aire.csv  | Dataset final con variables visuales incluidas      |

-------------------------------------------------------
Datos originales scrappeados (por barrio)
-------------------------------------------------------

- Idealista_dataset_arganzuela.csv
- Idealista_dataset_centro.csv
- Idealista_dataset_retiro.csv
- Idealista_dataset_chamberi.csv

-------------------------------------------------------
App
-------------------------------------------------------

El sistema se despliega como una aplicación web con Streamlit, disponible en el directorio:
`/app/`

Permite cargar propiedades, visualizar sus características y obtener el precio estimado
utilizando el modelo final entrenado.

-------------------------------------------------------
Autores
-------------------------------------------------------

- Álvaro Galán Romero
- Laura Andrés Cepas
- Ángel Sáenz Briones

TFM - Máster en Inteligencia Artificial y Big Data  
Universidad CEU San Pablo (2024–2025)
