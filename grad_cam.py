from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from torchvision.transforms import Compose, Normalize, ToTensor
import cv2
import os
import numpy as np
import torch
from matplotlib import pyplot as plt

#model = resnet50(pretrained=True)
#target_layers = [model.layer4[-1]]

def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def img_transform(img,isTensor=False):
    if not isTensor:
        rgb_img = cv2.imread(os.path.join(os.getcwd(),img), 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
    else:
        img=img.permute(1,2,0)
        rgb_img = cv2.resize(np.array(img), (224, 224))
        rgb_img = np.float32(np.array(img)) / 255
        
    return rgb_img

def heatmap(img,model,target_layers,target_category,cuda=False,isTensor=False):
    input_tensor=[]
    for img_name in img:
        input_tensor.append(preprocess_image(img_transform(img_name,isTensor), mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]))
    input_tensor=torch.cat(input_tensor, dim=0)    
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    return grayscale_cam
def show_cam_with_image(img,cam_output,rgb=True):
    visualization = show_cam_on_image(img_transform(img), cam_output, use_rgb=rgb)
    plt.imshow(visualization, interpolation='nearest')
    plt.show()
    return visualization
    






