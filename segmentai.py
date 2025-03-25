from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["default"](checkpoint=r"C:\Users\HOME\OneDrive - Universidad CES\Desktop\HGM\caja reveladora\segment-anything\checkpoints\sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = masks = mask_generator.generate(r"C:\Users\HOME\OneDrive - Universidad CES\Desktop\HGM\caja reveladora\src\IMG-20250310-WA0033.jpg")