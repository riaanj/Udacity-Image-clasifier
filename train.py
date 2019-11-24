# Import Libraries

import argparse
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
import torchvision.models as models
from collections import OrderedDict
import os

import json
# From matplotlib.pyplot import imshow


parser = argparse.ArgumentParser(description='Train the model')
parser.add_argument('data_directory',default='flowers',help="Directory where training images are stored")
parser.add_argument('--save_dir',default='./',dest='save_dir',help="Directory to save the checkpoint")
parser.add_argument('--learning_rate',default='0.001',dest='learning_rate',help="Directory to save the checkpoint (Default 0.001)",type=float)
parser.add_argument('--arch',default='vgg16',dest='arch',help="Select architechture (vgg16 or densenet121) (Default vgg16)")
parser.add_argument('--epochs',default='5',dest='epochs',help="Set amount of epochs to run training (Default 5)",type=int)
parser.add_argument('--hidden_units',default='512',dest='hidden_units',help="specify amount of hidden units to train with (Default 512)",type=int)
parser.add_argument('--gpu','--GPU',action='store_true',default=False,dest='gpu',help="Enable training with GPU (Default CPU)")
args = parser.parse_args()

# Change to either GPU or CPU. If GPU is selected, but not availiable, it will use the CPU but inform the user
if args.gpu:
    if torch.cuda.is_available():
        device='cuda'
    else:
        print("GPU is not availiable, using CPU")
        device='cpu'
else:
    device='cpu'

# Checks if ditectory for save_dir exists, if not it creates one if it has permission
if not os.path.isdir(args.save_dir):
    try:
        os.mkdir(args.save_dir)
    except PermissionError as e:
        print("No permission to create the directory: ",args.save_dir)
    except FileExistsError as e:
        print("Folder already exist, saving in ",args.save_dir)
    
   
            
    
# Data Directories
data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Transforms
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),                                      
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),                                     
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
validation_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),                                     
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# Load Data Sets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)

# Define Data Loaders
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=64)
validation_loader = torch.utils.data.DataLoader(validation_dataset,batch_size=64)

# Label Mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Build and train your network with either vgg16 or densenet121
if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    #define new Classifier
    classifier = nn.Sequential (OrderedDict ([
                            ('fc1',nn.Linear(25088,4096)),
                            ('relu1',nn.ReLU()),
                            ('dropout1',nn.Dropout(p=0.5)),
                            ('fc2',nn.Linear(4096,2056)),
                            ('relu2',nn.ReLU()),
                            ('fc3',nn.Linear(2056,args.hidden_units)),
                            ('relu3',nn.ReLU()),
                            ('dropout2',nn.Dropout(p=0.5)),
                            ('fc4',nn.Linear(args.hidden_units,102)),
                            ('output',nn.LogSoftmax(dim=1))
                            ]))

elif args.arch == 'densenet121':
    model =  models.densenet121(pretrained=True)
    # define new Classifier
    classifier = nn.Sequential (OrderedDict ([
                            ('fc1',nn.Linear(1024,1024)),
                            ('relu1',nn.ReLU()),
                            ('dropout1',nn.Dropout(p=0.5)),
                            ('fc2',nn.Linear(1024,512)),
                            ('relu2',nn.ReLU()),
                            ('fc3',nn.Linear(512,args.hidden_units)),
                            ('relu3',nn.ReLU()),
                            ('fc4',nn.Linear(args.hidden_units,102)),
                            ('output',nn.LogSoftmax(dim=1))
                            ]))
else:
    print("Model {} is not supported, please choose either vgg16 or densenet121".format(args.arch))
    exit(1)
# Dont train gradiants again for vgg
for p in model.parameters():
        p.requires_grad = False
# Replace Classifier
model.classifier = classifier



# Validation function

def validation(model,valloader,criterion,device):
    val_loss = 0
    accuracy = 0
    model.to(device)
    for images,labels in valloader:
        images,labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        val_loss += criterion(output,labels).item()
        
        propabilities = torch.exp(output)
        
        equality = (labels.data == propabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return val_loss,accuracy


criterion = nn.NLLLoss()
# Specify classifier to only train the classifier
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Set model to either GPU or CPU
model.to(device)
counter=0               # Counts each iteration
show_loss = 30          # Show the loss and accuracy every n iterations
epochs = args.epochs    # Go n times through the dataset to train, based on selection

print("Starting training on the "+device+" for "+str(args.epochs)+" epochs using architechture "+args.arch+".")
for epoch in range(epochs):
    
    running_loss = 0
    counter=0
    for images,labels in train_loader:
        model.train()                   # Make sure model is in training mode
        images = images.to(device)
        labels = labels.to(device)
        counter+=1       

        output = model.forward(images)  # Do a forward pass
        
        # Zero out the gradiant
        optimizer.zero_grad()
        
        loss = criterion(output,labels) # Calculate the loss 
        loss.backward()                 # Update the loss backward through the network
        optimizer.step()                #Perform a optimisation step
        
        running_loss +=loss.item()      # Returns a scalar value of the loss function
        
        if counter % show_loss == 0:
            model.eval()                # Switches to evalutation mode to save resources
            
            with torch.no_grad():       # Dont calculate gradients when evaluating the loss and accuracy
                v_loss,accuracy = validation(model,validation_loader,criterion ,device)            
            print("Epoch: {}/{} | ".format(epoch+1, epochs),
                  "Training Loss: {:.4f} | ".format(running_loss/show_loss),
                  "Validation Loss: {:.4f} | ".format(v_loss/len(validation_loader)),
                  "Validation Accuracy: {:.2f}%  |  ".format((accuracy/len(validation_loader))*100)                 
                 )
            model.train()               # Make sure model is in training mode again
            running_loss = 0            # Reset running loss value

# Save the checkpoint 
def save_checkpoint(model,file_path):
    
    model.class_to_idx = train_dataset.class_to_idx
    
    checkpoint = {'arch': args.arch,
                  'classifier' : model.classifier,
                 'class_to_idx': model.class_to_idx,
                 'model_state_dict' : model.state_dict(),
                  'Epochs done' : epoch
                 }
    torch.save(checkpoint,file_path)
    print("Saving model as: ",str(args.save_dir)+'/checkpoint.pth')
checkpoint = str(args.save_dir)+'/checkpoint.pth'
save_checkpoint(model,checkpoint)
