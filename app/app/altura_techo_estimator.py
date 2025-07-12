import streamlit as st

class SimpleCeilingHeightEstimator:
    def __init__(self):
        # Rangos típicos de altura de techo en Madrid
        self.height_ranges = {
            'low': (2.3, 2.6),      # Pisos modernos/reformados
            'medium': (2.6, 3.0),   # Estándar
            'high': (3.0, 3.8),     # Edificios clásicos con techos altos
            'very_high': (3.8, 4.5) # Edificios señoriales
        }

    def analyze_image_characteristics(self, image):
        """Analiza características visuales básicas de la imagen"""
        height, width, _ = image.shape

        # Convertir a escala de grises para análisis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 1. RATIO DE ASPECTO DE LA IMAGEN
        aspect_ratio = height / width

        # 2. DETECTAR LÍNEAS VERTICALES PROMINENTES (esquinas, marcos)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(height*0.3))

        vertical_lines = 0
        if lines is not None:
            for line in lines:
                _, theta = line[0]
                # Contar líneas verticales
                if abs(theta) < 0.3 or abs(theta - np.pi) < 0.3:
                    vertical_lines += 1

        # 3. ANALIZAR DISTRIBUCIÓN DE INTENSIDAD VERTICAL
        # Dividir imagen en franjas horizontales
        strip_height = height // 10
        brightness_profile = []

        for i in range(10):
            start_y = i * strip_height
            end_y = min((i + 1) * strip_height, height)
            strip = gray[start_y:end_y, :]
            avg_brightness = np.mean(strip)
            brightness_profile.append(avg_brightness)

        # 4. DETECTAR CARACTERÍSTICAS ARQUITECTÓNICAS
        # Buscar patrones que indiquen techos altos vs bajos

        # Variación de brillo en la parte superior (posible techo decorativo)
        upper_variation = np.std(brightness_profile[:3])

        # Contraste general de la imagen
        overall_contrast = np.std(gray)

        # Presencia de elementos en la parte superior de la imagen
        upper_third = gray[:height//3, :]
        upper_edges = cv2.Canny(upper_third, 50, 150)
        upper_edge_density = np.sum(upper_edges > 0) / upper_third.size

        return {
            'aspect_ratio': aspect_ratio,
            'vertical_lines': vertical_lines,
            'brightness_profile': brightness_profile,
            'upper_variation': upper_variation,
            'overall_contrast': overall_contrast,
            'upper_edge_density': upper_edge_density,
            'image_height': height,
            'image_width': width
        }

    def classify_ceiling_type(self, characteristics):
        """Clasifica el tipo de techo basándose en características visuales"""

        # Inicializar puntuaciones para cada tipo
        scores = {
            'low': 0,
            'medium': 0,
            'high': 0,
            'very_high': 0
        }

        # FACTOR 1: Ratio de aspecto de la imagen
        aspect_ratio = characteristics['aspect_ratio']
        if aspect_ratio < 0.6:  # Imagen muy ancha = vista panorámica = techo bajo
            scores['low'] += 2
            scores['medium'] += 1
        elif aspect_ratio < 0.8:  # Imagen moderadamente ancha
            scores['medium'] += 2
            scores['low'] += 1
        elif aspect_ratio < 1.2:  # Imagen cuadrada
            scores['medium'] += 2
            scores['high'] += 1
        else:  # Imagen alta = posible techo alto
            scores['high'] += 2
            scores['very_high'] += 1
            if aspect_ratio > 1.5:
                scores['very_high'] += 1

        # FACTOR 2: Líneas verticales (elementos arquitectónicos)
        vertical_lines = characteristics['vertical_lines']
        if vertical_lines > 8:  # Muchas líneas = arquitectura compleja = techo alto
            scores['high'] += 2
            scores['very_high'] += 1
        elif vertical_lines > 4:
            scores['medium'] += 1
            scores['high'] += 1
        else:
            scores['low'] += 1
            scores['medium'] += 1

        # FACTOR 3: Variación en la parte superior (molduras, decoración)
        upper_variation = characteristics['upper_variation']
        if upper_variation > 30:  # Mucha variación = decoración = techo alto
            scores['high'] += 2
            scores['very_high'] += 1
        elif upper_variation > 15:
            scores['medium'] += 1
            scores['high'] += 1
        else:
            scores['low'] += 1
            scores['medium'] += 1

        # FACTOR 4: Densidad de bordes en parte superior
        upper_edge_density = characteristics['upper_edge_density']
        if upper_edge_density > 0.1:  # Muchos detalles arriba = techo decorado
            scores['high'] += 1
            scores['very_high'] += 1
        elif upper_edge_density > 0.05:
            scores['medium'] += 1
        else:
            scores['low'] += 1

        # FACTOR 5: Contraste general
        overall_contrast = characteristics['overall_contrast']
        if overall_contrast > 60:  # Alto contraste = espacios amplios = techos altos
            scores['high'] += 1
            scores['very_high'] += 1
        elif overall_contrast > 40:
            scores['medium'] += 1
        else:
            scores['low'] += 1

        # Encontrar la categoría con mayor puntuación
        best_category = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_category] / sum(scores.values())

        return best_category, confidence, scores

    def estimate_height_from_category(self, category, confidence):
        """Convierte la categoría en una altura específica"""
        range_min, range_max = self.height_ranges[category]

        # Usar un valor dentro del rango, ajustado por confianza
        if confidence > 0.7:
            # Alta confianza: usar valor medio del rango
            height = (range_min + range_max) / 2
        elif confidence > 0.5:
            # Confianza media: tender hacia el centro general (2.7m)
            height = (range_min + range_max) / 2
            height = height * 0.7 + 2.7 * 0.3  # Mezclar con valor conservador
        else:
            # Baja confianza: usar valor conservador
            height = 2.7

        # Añadir pequeña variación aleatoria para evitar valores idénticos
        height += np.random.normal(0, 0.05)  # ±5cm de variación

        return max(2.2, min(4.5, height))  # Limitar entre 2.2 y 4.5m

    def analyze_single_image(self, image_path):
        """Analiza una sola imagen"""
        try:
            # Cargar imagen
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"No se pudo cargar la imagen: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path

            # Analizar características
            characteristics = self.analyze_image_characteristics(image)

            # Clasificar tipo de techo
            category, confidence, scores = self.classify_ceiling_type(characteristics)

            # Estimar altura
            estimated_height = self.estimate_height_from_category(category, confidence)

            return {
                'estimated_height': round(estimated_height, 2),
                'category': category,
                'confidence': round(confidence, 2),
                'scores': scores,
                'characteristics': characteristics
            }

        except Exception as e:
            print(f"Error analizando imagen {image_path}: {e}")
            return {
                'estimated_height': 2.7,  # Valor por defecto
                'category': 'medium',
                'confidence': 0.2,
                'error': str(e)
            }

    def analyze_property_images(self, image_folder, property_id):
        """Analiza todas las imágenes de una propiedad"""
        try:
            # Buscar imágenes de la propiedad
            all_files = os.listdir(image_folder)
            property_files = [f for f in all_files if f.startswith(f'foto_{property_id}_') and f.lower().endswith('.webp')]

            if not property_files:
                return {
                    'property_id': property_id,
                    'estimated_height': 2.7,
                    'confidence': 0.2,
                    'images_analyzed': 0,
                    'method': 'no_images_found'
                }

            # Analizar cada imagen
            results = []
            for filename in property_files:
                image_path = os.path.join(image_folder, filename)
                result = self.analyze_single_image(image_path)
                result['filename'] = filename
                results.append(result)

            # Combinar resultados
            valid_results = [r for r in results if 'error' not in r]

            if not valid_results:
                return {
                    'property_id': property_id,
                    'estimated_height': 2.7,
                    'confidence': 0.2,
                    'images_analyzed': len(results),
                    'method': 'all_errors'
                }

            # Calcular estadísticas
            heights = [r['estimated_height'] for r in valid_results]
            confidences = [r['confidence'] for r in valid_results]
            categories = [r['category'] for r in valid_results]

            # Altura final: mediana (más robusta que media)
            final_height = np.median(heights)

            # Confianza final: promedio ponderado
            final_confidence = np.mean(confidences)

            # Ajustar confianza según consistencia
            height_std = np.std(heights)
            if height_std < 0.2:  # Muy consistente
                final_confidence = min(0.9, final_confidence + 0.2)
            elif height_std < 0.4:  # Razonablemente consistente
                final_confidence = final_confidence
            else:  # Inconsistente
                final_confidence = max(0.3, final_confidence - 0.2)

            # Categoría más común
            most_common_category = max(set(categories), key=categories.count)

            return {
                'property_id': property_id,
                'estimated_height': round(final_height, 2),
                'confidence': round(final_confidence, 2),
                'images_analyzed': len(valid_results),
                'height_std': round(height_std, 2),
                'most_common_category': most_common_category,
                'method': 'multi_image_median',
                'individual_results': results
            }

        except Exception as e:
            print(f"Error procesando propiedad {property_id}: {e}")
            return {
                'property_id': property_id,
                'estimated_height': 2.7,
                'confidence': 0.2,
                'images_analyzed': 0,
                'method': 'error',
                'error': str(e)
            }
            
import numpy as np
import cv2
from PIL import Image
from collections import Counter

def bytes_to_image(bytes_img):
    # Convierte bytes a array numpy para OpenCV
    nparr = np.frombuffer(bytes_img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB
    return img

def estimate_ceiling_height(image_list):
    logs = []
    def log(msg):
        logs.append(msg)

    log("🔍 Iniciando estimación de altura de techo...")

    if not image_list:
        log("⚠️ No hay imágenes para analizar. Retornando valor por defecto 2.7")
        return 2.7, logs

    log(f" Número de imágenes recibidas: {len(image_list)}")

    estimator = SimpleCeilingHeightEstimator()
    results = []
	
    for i, img_bytes in enumerate(image_list):
    	st.write(f"Procesando imagen {i}")
    	img = bytes_to_image(img_bytes)
    	if img is None:
    		st.write(f"⚠️ Imagen {i} no pudo ser decodificada. Se ignora.")
    		continue
    	result = estimator.analyze_single_image(img)
    	st.write(f"Resultado imagen {i}: {result}")
    	results.append(result)
    heights = [r['estimated_height'] for r in results if 'error' not in r]
    st.write(f"Alturas obtenidas: {heights}")
    if not heights:
    	st.write("⚠️ No se obtuvieron alturas válidas, devolviendo valor por defecto 2.7")
    	final_height = 2.7
    else:
    	final_height = np.median(heights)
    	st.write(f"Altura final (mediana): {final_height}")

    return estimated_height, logs

