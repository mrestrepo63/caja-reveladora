import cv2
import numpy as np
import os

# Lista de imágenes en la carpeta 'src'
image_files = [f for f in os.listdir('src') if f.startswith('IMG-20250310-WA') and f.endswith('.jpg')]

for image_file in image_files:
    # Cargar la imagen
    imagen = cv2.imread(os.path.join('src', image_file))

    # Convertir a HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Crear una máscara para detectar tonos de piel y fluorescencia
    lower_skin = np.array([0, 0, 20], dtype=np.uint8)  
    upper_skin = np.array([190, 255, 200], dtype=np.uint8)  
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Operaciones morfológicas para reducir ruido
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Aplicar la máscara y convertir a escala de grises
    hand_only = cv2.bitwise_and(imagen, imagen, mask=mask)
    gray_hand = cv2.cvtColor(hand_only, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro bilateral para reducir ruido sin perder bordes
    filtered = cv2.bilateralFilter(gray_hand, 9, 75, 75)

    # Aplicar detección de bordes con Canny
    edges = cv2.Canny(filtered, 50, 150)

    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos pequeños para evitar ruido
    min_contour_area = 300  
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Dibujar contornos en la imagen original
    contornos_img = imagen.copy()
    cv2.drawContours(contornos_img, filtered_contours, -1, (0, 255, 0), 2)

    # Guardar imágenes procesadas
    cv2.imwrite(f'contornos_mano_{image_file}', contornos_img)
    cv2.imwrite(f'edges_{image_file}', edges)

cv2.destroyAllWindows()
