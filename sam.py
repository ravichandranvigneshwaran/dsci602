# Import necessary libraries
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
import os

# Set environment variable to avoid duplicate library errors
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Additional imports
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
import torchvision

# Print PyTorch and Torchvision versions, and check if CUDA is available
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

# Function to visualize annotations on an image
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Create an image with an alpha channel for transparency
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    
    # Color the image based on the annotations
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

# Set the SAM checkpoint file and model type
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

# Set the device to CUDA if available, else use CPU
device = "cuda"

# Load the SAM model and move it to the specified device
sam = sam_model_registry[model_type](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device=device)

# Create a mask generator using the SAM model
mask_generator = SamAutomaticMaskGenerator(sam)

# Read an image and convert it to RGB
img = cv2.imread("hum2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Generate masks using the mask generator
masks = mask_generator.generate(img)

# Plot the original image along with the generated masks
plt.figure(figsize=(20, 20))
plt.imshow(img)
show_anns(masks)
plt.axis('off')
plt.show()
