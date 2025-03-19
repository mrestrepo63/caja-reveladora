import cv2
import numpy as np
import ctypes
from pixy2 import Pixy2CCC

# Inicializar Pixy2 en Windows
pixy2 = Pixy2CCC()
pixy2.init()
pixy2.change_prog("video")

# Definir buffer para almacenar la imagen de Pixy2 en Windows
frame_buffer = (ctypes.c_uint8 * (320 * 200))()
frame_pointer = ctypes.cast(frame_buffer, ctypes.POINTER(ctypes.c_uint8))  # Necesario en Windows

# Función para capturar frames desde Pixy2
def get_frame():
    result = pixy2.video_get_frame(0, 0, 320, 200, frame_pointer)
    if result == 0:  # Si la captura fue exitosa
        return np.ctypeslib.as_array(frame_pointer, shape=(200, 320))  # Convertir a array numpy
    return None

while True:
    # Capturar imagen desde Pixy2
    frame = get_frame()
    if frame is None:
        continue  # Si no hay imagen, sigue intentando

    # Convertir a escala de grises y aplicar ecualización del histograma
    gray_hand = cv2.equalizeHist(frame)

    # Aplicar desenfoque gaussiano para reducir ruido
    blurred = cv2.GaussianBlur(gray_hand, (5, 5), 0)

    # Aplicar umbral para detectar áreas fluorescentes
    _, fluorescence = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)

    # Encontrar contornos de la fluorescencia
    contours, _ = cv2.findContours(fluorescence, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convertir a BGR para visualizar los contornos en color
    imagen_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(imagen_color, contours, -1, (0, 255, 0), 2)

    # Mostrar resultados
    cv2.imshow('Silueta de la Mano', gray_hand)
    cv2.imshow('Detección de Fluorescencia', fluorescence)
    cv2.imshow('Contornos de Fluorescencia', imagen_color)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
