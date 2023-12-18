# Import necessary libraries
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import numpy as np
import os, json
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn.functional as F

# Function to get image from the file path
def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB') 

# Load and display an image
img = get_image("hum2.jpg")
plt.imshow(img)
plt.show()

# Function to get the transformation for input image
def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])       
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])    

    return transf

# Function to get input tensors for the model
def get_input_tensors(img):
    transf = get_input_transform()
    # unsqueeze converts single image to batch of 1
    return transf(img).unsqueeze(0)

# Load pre-trained model and processor for image classification
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
resnet50 = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
resnet50.eval()

# Process the input image using the processor and get model predictions
inputs = processor(img, return_tensors="pt")
with torch.no_grad():
    logits = resnet50(**inputs).logits

# Calculate class probabilities using softmax and print top 5 predictions
probs = F.softmax(logits, dim=1)
probs5 = probs.topk(5)
print(tuple((p, c, resnet50.config.id2label[c]) for p, c in zip(probs5[0][0].detach().numpy(), probs5[1][0].detach().numpy())))

# Function to get PIL transformation
def get_pil_transform(): 
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])    

    return transf

# Function to get preprocess transformation
def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])     
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf    

# Apply transformations to the image
pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

# Function to predict on a batch of images
def batch_predict(images):
    resnet50.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet50.to(device)
    batch = batch.to(device)
    
    logits = resnet50(batch).logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

# Predict on the test image using the batch_predict function
test_pred = batch_predict([pill_transf(img)])
test_pred.squeeze().argmax()

# Use LIME to explain the model predictions
from lime import lime_image

explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(pill_transf(img)), 
                                         batch_predict, # classification function
                                         top_labels=5, 
                                         hide_color=0, 
                                         num_samples=1000) # number of images that will be sent to the classification function

# Visualize LIME explanations
from skimage.segmentation import mark_boundaries

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
img_boundry1 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry1)
plt.show()

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
img_boundry2 = mark_boundaries(temp/255.0, mask)
plt.imshow(im
