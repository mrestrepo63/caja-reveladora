import numpy as np
import cv2
import torch
import depthai as dai
import os
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

print("Algoritmo de video a imágenes")
print("CUDA is available:", torch.cuda.is_available())

# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1)

# Inicializar el modelo SAM
# sam_checkpoint = "src/models/sam_vit_l.pth"
# model_type = "vit_l"
# device = "cuda" if torch.cuda.is_available() else "cpu"
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
# predictor = SamPredictor(sam)

# # Función para segmentar una imagen capturada
# def segment(image):
#     image_height, image_width = image.shape[:2]

#     # Configurar la imagen en el predictor
#     predictor.set_image(image)

#     # Definir punto de referencia para segmentación (centro de la imagen)
#     input_point = np.array([[image_width / 2, image_height / 2]])
#     input_label = np.array([1])

#     # Obtener máscaras del modelo
#     masks, _, _ = predictor.predict(
#         point_coords=input_point,
#         point_labels=input_label,
#         multimask_output=True,
#     )

#     # Aplicar máscara a la imagen
#     segmented_image = image.copy()
#     segmented_image[masks[0] == False] = [255, 255, 255]

#     return segmented_image

# Crear pipeline de DepthAI para capturar imágenes desde OAK-1
pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)
# cam_rgb.setPreviewSize(1920, 1080)  # Ajustar resolución
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
xout_video = pipeline.create(dai.node.XLinkOut)
xout_video.setStreamName("video")
cam_rgb.preview.link(xout_video.input)

# Iniciar cámara OAK-1 y procesar imágenes en tiempo real
# Crear directorio para frames
os.makedirs('frames', exist_ok=True)
frame_count = 0

with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=1, blocking=False)

    while True:
        frame_data = video_queue.get()
        frame = frame_data.getCvFrame()  # Convertir a imagen de OpenCV

        # Guardar frame como imagen
        cv2.imwrite(f'frames/frame_{frame_count:04d}.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_count += 1

        # Mostrar imágenes
        cv2.imshow("OAK-1 Live Feed", frame)
        # cv2.imshow("Segmented Image", segmented_image)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar recursos
cv2.destroyAllWindows()
