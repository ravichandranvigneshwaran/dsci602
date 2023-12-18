from __future__ import print_function, division
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import urllib
from torchvision import transforms
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import argparse
import glob
import os
from sklearn import metrics
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
from skimage.segmentation import mark_boundaries
from sklearn.linear_model import LinearRegression
import cv2

def generate_concept_masks(gen_model,target_img):
    return gen_model.generate(target_img)

def calculateShap(model,img_numpy,image_class,concept_masks,fc,feat_exp,image_norm=None,lr=0.008):
    
    
    feat, probs,losses,net,bin_x_torch = PIE(feat_exp,model,concept_masks,img_numpy,image_class,fc,lr=lr,epochs=100,image_norm=image_norm) ### learning the PIE module 
    feat_num = len(concept_masks)
    shap_val = []
    mc = 50000
    
    
    for i in range(feat_num):
        bin_x_tmp = np.random.binomial(1,0.5,size=(mc,feat_num)) ### mc shapley computing 
        bin_x_tmp_sec = bin_x_tmp.copy()

        bin_x_tmp[:,i] = 1
        bin_x_tmp_sec[:,i] = 0

        bin_x_tmp = torch.from_numpy(bin_x_tmp).type(torch.float)
        bin_x_tmp_sec = torch.from_numpy(bin_x_tmp_sec).type(torch.float)

        pre_shap = (feature_prob(fc,net.forward_feat(bin_x_tmp.cuda()),image_class) - feature_prob(fc,net.forward_feat(bin_x_tmp_sec.cuda()),image_class)).detach().cpu().numpy()

        shap_val.append(pre_shap.sum()/mc)
    ans = shap_val.index(max(shap_val))
    shap_list = shap_val

    shap_list = np.array(shap_list)
    shap_arg = np.argsort(-shap_list)
    auc_mask = np.expand_dims(np.array([concept_masks[shap_arg[:i+1]].sum(0) for i in range(len(shap_arg))]).astype(bool),3)


    return auc_mask, shap_list

def feature_prob(model,feat,target_label):
    
    with torch.no_grad():
        if model == None:
            probabilities = torch.nn.functional.softmax(feat,dim=1)
        else:
            output = model(feat)
            probabilities = torch.nn.functional.softmax(output,dim=1)
        return probabilities[:,target_label]
    
class NeuralNet(nn.Module): ### a simple NN network
    def __init__(self, in_size,bs,head,lr,ftdim):
        super(NeuralNet, self).__init__()
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(in_size,ftdim))
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.loss = torch.nn.BCELoss()
        self.bs = bs
        self.fc = head

    def change_lr(self,lr):
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

    def forward(self,x):
        if self.fc != None:
            return self.fc(self.model(x))
        else:
            return self.model(x)
    
    def forward_feat(self,x):
        return self.model(x)
    
    def step(self, x,y):
        self.optimizer.zero_grad()
        loss = self.loss
        output = loss(x, y)
        output.backward()
        self.optimizer.step()
        return output.detach().cpu().numpy()
    
    def step_val(self, x,y):
        self.optimizer.zero_grad()
        loss = self.loss
        output = loss(x, y)
        return output.detach().cpu().numpy()


def learning_feature(target_model,full_model,concept_mask,target_img,target_label,fc,image_norm=None):
    target_img = np.expand_dims(target_img,0)
    concept_mask = np.expand_dims(concept_mask,3)
    masked_image = target_img*concept_mask
    batch_img = []
    for i in range(masked_image.shape[0]):
        input_tensor = image_norm(masked_image[i])
        batch_img.append((input_tensor.cpu()))
    tmp_dl = DataLoader(dataset = batch_img, batch_size = 500, shuffle =False)
    
    output = None
    fc_res = None
    for x in tmp_dl:
        with torch.no_grad():
            if output == None:
                output = target_model(x.cuda())
                output = output['avgpool'].squeeze().squeeze()
                fc_res = torch.nn.functional.softmax(fc(output), dim=1)[:,target_label]
            else:
                tmp_out = target_model(x.cuda())
                tmp_out = tmp_out['avgpool'].squeeze().squeeze()
                output = torch.cat((output,tmp_out))
                fc_res = torch.cat((fc_res,torch.nn.functional.softmax(fc(tmp_out), dim=1)[:,target_label]))
        
    return output, fc_res

    
def PIE(target_model,full_model,concept_mask,target_img,target_label,fc,lr,epochs,image_norm=None):
    masks_tmp = concept_mask.copy()
    
    num_feat = masks_tmp.shape[0]
    only_feat = np.zeros((num_feat, num_feat), int)
    np.fill_diagonal(only_feat,1)
    bin_x = np.random.binomial(1,0.5,size=(2500,num_feat)).astype(bool) #### generate 2500 samples to learn PIE by default
    new_mask = np.array([masks_tmp[i].sum(0) for i in bin_x]).astype(bool)
    feat, probs = learning_feature(target_model,full_model,new_mask,target_img,target_label,fc,image_norm)
    feat = feat.detach().clone().cpu()
    probs = probs.detach().clone().cpu()
    bin_x_torch = torch.tensor(bin_x.tolist(),dtype=torch.float)
    data = [[x,y] for x,y in zip(bin_x_torch,probs)]
    bs = 100
    losses = []
    
    net = NeuralNet(num_feat,bs,fc,lr,feat.shape[1]).cuda()
    
    net.change_lr(lr)
    data_comb_train = DataLoader(dataset = data[num_feat:], batch_size = bs, shuffle =True)
    
    ##### learning combin
    for epoch in range(50):
        loss = 0
        for x,y in data_comb_train:
            pred = torch.nn.functional.softmax(net(x.cuda()), dim=1)[:,target_label]
            loss += net.step(pred,y.cuda())*x.shape[0]
    
    net.eval()
    return feat, probs,losses,net,bin_x_torch


# Define the target model
model = models.resnet50(weights='IMAGENET1K_V2').cuda()
cvmodel = model.cuda()
cvmodel.eval()

# Define the feature extractor
feat_exp = create_feature_extractor(cvmodel, return_nodes=['avgpool'])
fc = model.fc
model.eval()
feat_exp.eval()
cvmodel.eval()

# Define the SAM model and mask generator
sam = sam_model_registry["default"](checkpoint="./sam_vit_h_4b8939.pth")
sam.to("cuda")
mask_generator = SamAutomaticMaskGenerator(sam)

# Define three ways to pre-process a given image
test_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_reshape = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])

image_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the test image
img = Image.open('image/dog.jpg').convert('RGB')

# Predict the original image class
predict_org = torch.nn.functional.softmax(model(test_preprocess(img).unsqueeze(0).cuda()), dim=1)
pred_image_class = int(torch.argmax(predict_org))

# Generate masks using SAM
for_mask_image = np.array(image_reshape(img))
input_image_copy = for_mask_image.copy()
org_masks = generate_concept_masks(mask_generator, input_image_copy)
concept_masks = np.array([i['segmentation'].tolist() for i in org_masks])

# Compute calculateShap
auc_mask, shap_list = calculateShap(model, input_image_copy, pred_image_class, concept_masks, fc, feat_exp, image_norm=image_norm)

# Select the concept patch with the highest shapley value
final_explain = (for_mask_image * auc_mask[0])
final_explain = Image.fromarray(final_explain)

# Save the final explanation output
final_explain.save("hum.jpeg")
