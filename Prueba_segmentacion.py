# from segment import SamPredictor, sam_model_registry
# sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
# predictor = SamPredictor(sam)
# predictor.set_image()
# masks, _, _ = predictor.predict(<input_prompts>)

import sys
import os
import matplotlib.pyplot as plt
import torch 
import numpy as np

sys.path.append(os.path.abspath("segment-anything"))
from segment_anything import sam_model_registry

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = r"C:\Users\palac\Downloads\sam_vit_h_4b8939.pth"
model_type = "default"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

import cv2

# Cargar la imagen
image = cv2.imread(r"test\Imagenes fluorescencia\IMG-20250310-WA0002.jpg")

# Asegurar que la imagen se cargó correctamente
if image is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)




from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["default"](checkpoint=r"C:\Users\palac\Downloads\sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
mask1=mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 