# Import Libraries

import argparse
import json
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
import torchvision.models as models
from collections import OrderedDict

from PIL import Image
import os,glob

# Parser section
parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('photo',help="Path to flower image")
parser.add_argument('checkpoint',help="Checkpoint to load")
parser.add_argument('--topk',default='1',dest='topk',help="Top k amount of results to return (Default 1)",type=int)
parser.add_argument('--category_names',dest='category_names',default='cat_to_name.json',help="The category names file (Default cat_to_name.json) ")
parser.add_argument('--gpu','--GPU',action='store_true',default=False,dest='gpu',help="Enable predictions with GPU (Default CPU)")
args = parser.parse_args()

if torch.cuda.is_available() and args.gpu == True:
    device='cuda'
else:
    device='cpu'
    
# function that loads a checkpoint and rebuilds the model
def load_checkpoint(file_path):
    if device == 'cuda':
        checkpoint = torch.load(file_path)
    else:
        checkpoint = torch.load(file_path,'cpu')
    print("Loading checkpoint {} with architecture: {}. Doing prediction with {}".format(file_path,checkpoint['arch'],device) )
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['model_state_dict'])
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        return model
    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = checkpoint['classifier']
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['model_state_dict'])
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        return model
    else:
        print("Architechture is not supported")

model = load_checkpoint(args.checkpoint)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
        
    mean = [0.485,0.456,0.406]          # Values for the mean normilisation
    std = [0.229, 0.224, 0.225]         # Values for the standard deviation normilisation
 
    # Opens the image and resizes it 
    im = Image.open(image)
    if im.size[0] > im.size[1]:
        size = im.size[0],256
    else:
        size = 256,im.size[1]
    im.thumbnail(size)  
    
    # Center crop the image
    left = ((im.size[0]-244)/2)
    upper =  ((im.size[1]-244)/2)
    right = im.size[0] - ((im.size[0]-244)/2)
    lower = im.size[1] - ((im.size[1]-244)/2)
    im = im.crop((left,upper,right,lower))
       
    # Convert image to numpy array    
    np_image = np.array(im)
    
    # Change the value to between 0 and 1 and normilise the image
    np_image = np.divide(np_image,255)
    np_image = np.divide(np.subtract(np_image,mean),std)
    np_image = np.transpose(np_image,(2,0,1))
 
    return np_image
    
    
def predict(image_path, model, topk=args.topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)                                            # Change model to CPU or GPU
    image = process_image(image_path)                           # Process the image
    if device == 'cuda':
        image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    else:
        image = torch.from_numpy(image).type(torch.FloatTensor)
    
    image = image.unsqueeze(0)                                  # Flatten image
    image.to(device)        
    with torch.no_grad():
        output = model.forward(image)                               # Put image throught model
        
    probabilities = torch.exp(output)                           # Get the propabilities for a class
    top_prob,top_idx = torch.topk(probabilities,topk)           # Get top k propabilities
    
    # Convert to lists
    top_prob = top_prob.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_idx =top_idx.detach().type(torch.FloatTensor).numpy().tolist()[0]
    
    idx_to_class = {value:key for key,value in model.class_to_idx.items()}
    
    top_class = [idx_to_class[index] for index in top_idx]
    
    # Open and read the catagory to names file
    with open('cat_to_name.json', 'r') as file:
        cat_to_name = json.load(file)
    # Put names to the top catagories
    top_names = [cat_to_name[i] for i in top_class]
    return top_prob,top_names

# Prints probability for image
probability,flower_name = predict(args.photo,model)
print("Flower name and probability")
for i in range(len(probability)):
    print(flower_name[i], "  ",round(probability[i]*100,2))
    
