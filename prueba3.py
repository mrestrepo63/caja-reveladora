import cv2
import numpy as np
import os

# Lista de imagenes en la carpeta 'src'
image_files = [f for f in os.listdir('src') if f.startswith('IMG-20250310-WA') and f.endswith('.jpg')]

for image_file in image_files:
    # Cargar la imagen
    imagen = cv2.imread(os.path.join('src', image_file))

    # Convertir a escala de grises para detectar intensidades
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar ecualización del histograma para mejorar el contraste
    gray = cv2.equalizeHist(gray)

    # Aplicar un desenfoque gaussiano para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5),0)

    # Aplicar un umbral adaptativo para segmentar mejor las áreas iluminadas
    threshold = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Encontrar contornos
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar contornos en la imagen original
    cv2.drawContours(imagen, contours, -1, (0, 255, 0), 2)

    # Mostrar la imagen con los contornos detectados
    cv2.namedWindow('Contornos de la Mano', cv2.WINDOW_NORMAL)
    cv2.imshow('Contornos de la Mano', imagen)

    cv2.namedWindow('Umbral', cv2.WINDOW_NORMAL)
    cv2.imshow('Umbral', threshold)

    # Esperar una tecla antes de procesar la siguiente imagen
    cv2.waitKey(0)

cv2.destroyAllWindows()
