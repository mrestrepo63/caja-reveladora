import cv2
import numpy as np
import os

# Lista de imagenes en la carpeta 'src'
image_files = [f for f in os.listdir('src') if f.startswith('IMG-20250310-WA') and f.endswith('.jpg')]

for image_file in image_files:
    # Cargar la imagen
    imagen = cv2.imread(os.path.join('src', image_file))

    # Convertir a espacio de color HSV para aislar la mano del fondo
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Crear una máscara para la mano (detectando tonos más brillantes)
    lower_skin = np.array([0, 0, 50], dtype=np.uint8)
    upper_skin = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Aplicar operaciones morfológicas para mejorar la segmentación
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Aplicar la máscara a la imagen original para obtener la silueta de la mano
    hand_only = cv2.bitwise_and(imagen, imagen, mask=mask)

    # Convertir la imagen de la mano a escala de grises
    gray = cv2.cvtColor(hand_only, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral para detectar áreas de interés
    _, threshold = cv2.threshold(gray, 100, 200, cv2.THRESH_BINARY)

    # Encontrar contornos
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar contornos en la imagen original
    cv2.drawContours(imagen, contours, -1, (0, 255, 0), 3)

    # Mostrar la imagen con los contornos en su tamaño original
    cv2.namedWindow('Contornos de la Mano', cv2.WINDOW_NORMAL)  # Permitir redimensionar la ventana
    cv2.imshow('Contornos de la Mano', imagen)

    cv2.namedWindow('Umbral', cv2.WINDOW_NORMAL)  # Permitir redimensionar la ventana
    cv2.imshow('Umbral', threshold)

    cv2.namedWindow('Silueta de la Mano', cv2.WINDOW_NORMAL)
    cv2.imshow('Silueta de la Mano', gray)

    # Esperar una tecla antes de pasar a la siguiente imagen
    cv2.waitKey(0)

cv2.destroyAllWindows()