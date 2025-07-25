import os
os.environ["TRANSFORMERS_NO_JAX"] = "1" 
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch
import numpy as np

# Load DINOv2 base once
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")
model.eval()

def extract_dino_feature(image: Image.Image) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [1, 768]
    return cls_token.squeeze().numpy()