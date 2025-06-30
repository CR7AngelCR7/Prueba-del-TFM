# Prueba-del-TFM

# Predicción de Precios de Propiedades en Madrid

Este proyecto predice precios de propiedades en Madrid combinando datos scrappeados, características geográficas y análisis de imágenes.

## Notebooks

### `1get_dataset.ipynb`
- Une datasets (barrios, EDA, imputaciones).
- Limpieza y carga de imágenes.
- Archivos generados:
  - `datos.csv`: Datos originales scrappeados + eda .
  - `datos_promedio.csv`: Versión sin outliers en precio.

### `2AutoML_pruebas.ipynb`
- Pruebas con PyCaret, LazyPredict, AutoML.
- Comparación de modelos y ensembler.
- No genera nuevos archivos.

### `3_ModeloSinFotos.ipynb`
- EDA por barrios y mapas.
- Cálculo de distancia a metro.
- Modelado sin imágenes.
- Archivos generados:
  - `datos_limpiozonas.csv`: Limpio y sin outliers por barrio.
  - `datos_limpiometro.csv`: Con distancia al metro más cercano.

## `4_ModeloConFotos.ipynb`
- Estimación de altura de techos con imágenes.
- Clasificación de viviendas reformadas.
- Modelado con variables visuales.
- Archivo generado:
  - `datos_alturatechos.csv`: Dataset con columna `altura_techo`.

## CSVs – Descripción

| Archivo                   | Contenido                                                  |
|---------------------------|------------------------------------------------------------|
| `datos.csv`               | Datos originales scrappeados + eda, imputaciones.          |
| `datos_promedio.csv`      | Datos sin outliers de precio.                              |
| `datos_limpiozonas.csv`   | Datos limpios por barrio sin outliers.                     |
| `datos_limpiometro.csv`   | Datos con distancia al metro más cercano.                  |
| `datos_alturatechos.csv`  | Datos con altura de techos estimada por análisis de imágenes.|

## Requisitos
- Python 3.8+
- Jupyter Notebook
- Bibliotecas: pandas, numpy, scikit-learn, pycaret, lazypredict, PIL, etc.
