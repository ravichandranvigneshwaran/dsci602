# Import necessary libraries
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import os

# Set an environment variable to disable ONEDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load a dataset from the Hugging Face library
# Note: Make sure to run 'huggingface-cli login' if the dataset is gated/private
dataset = load_dataset("zh-plus/tiny-imagenet")

# Read an image using OpenCV
image = cv2.imread("dog.jpg")

# Load pre-trained image processor and model from Hugging Face
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# Print the configuration of the loaded model
print(model.config)

# Process the input image using the image processor
inputs = processor(image, return_tensors="pt")

# Make predictions using the loaded model
with torch.no_grad():
    logits = model(**inputs).logits

# Get the predicted label by selecting the index with the highest probability
predicted_label = logits.argmax(-1).item()

# Print the label corresponding to the predicted index
print(model.config.id2label[predicted_label])
