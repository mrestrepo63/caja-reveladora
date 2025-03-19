import cv2
import numpy as np
import os

# Lista de imágenes en la carpeta 'src'
image_files = [f for f in os.listdir('src') if f.startswith('images') and f.endswith('.jpeg')]

for image_file in image_files:
    # Cargar la imagen
    imagen = cv2.imread(os.path.join('src', image_file))

    # Convertir a espacio de color HSV para aislar la mano del fondo
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Crear una máscara para la mano (detectando tonos más brillantes)
    lower_skin = np.array([0, 0, 5], dtype=np.uint8)  # Detectar tonos brillantes
    upper_skin = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Aplicar operaciones morfológicas para mejorar la segmentación y reducir ruido
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Elimina pequeñas partículas de ruido
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Une áreas segmentadas

    # Aplicar la máscara a la imagen original para obtener solo la mano
    hand_only = cv2.bitwise_and(imagen, imagen, mask=mask)

    # Convertir a escala de grises
    gray_hand = cv2.cvtColor(hand_only, cv2.COLOR_BGR2GRAY)

    # Aplicar ecualización del histograma para mejorar el contraste
    gray_hand = cv2.equalizeHist(gray_hand)

    # Aplicar un desenfoque gaussiano para reducir ruido
    blurred = cv2.GaussianBlur(gray_hand, (5, 5), 0)

    # Aplicar un umbral para detectar las áreas fluorescentes
    _, fluorescence = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)

    # Encontrar contornos de la fluorescencia
    contours, _ = cv2.findContours(fluorescence, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos pequeños para evitar ruido
    min_contour_area = 50  
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Dibujar los contornos en la imagen original
    cv2.drawContours(imagen, filtered_contours, -1, (0, 255, 0), 2)

    # Mostrar resultados
    # cv2.namedWindow('Silueta de la Mano', cv2.WINDOW_NORMAL)
    cv2.imwrite('silueta_mano.png', gray_hand)

    # cv2.namedWindow('Detección de Negros Y Blancos', cv2.WINDOW_NORMAL)
    cv2.imwrite('deteccion_BW.png', fluorescence)

    # cv2.namedWindow('Contornos de Fluorescencia', cv2.WINDOW_NORMAL)
    cv2.imwrite('contornos.png', imagen)

    # Esperar una tecla antes de procesar la siguiente imagen
    cv2.waitKey(0)

cv2.destroyAllWindows()
