# Licence : BSD

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import pandas as pd

# Data augmentation and Normalization for training
# Just Normalization for Validation

data_transforms = {
    'train': transforms.Compose([ # Composes several transforms together
    transforms.RandomResizedCrop(224), # Crop the given PIL.Image to random size and aspect ratio
    transforms.RandomHorizontalFlip(), # Horizontally flips the given PIL-Image with a probability of 0.5
    transforms.ToTensor(), # Convert a PIL.Image or numpy.ndarray to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Normalize an tensor image with mean and standard deviation
]),
'val': transforms.Compose([
    transforms.Resize(256), # Rescale the input PIL.image into desired size
    transforms.CenterCrop(224), # Crops the PIL.Image at the center
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
}

data_dir = '/mnt/disks/dataset'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                  data_transforms[x]) for x in ['train', 'val']}
data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128, shuffle=True,
                num_workers=8) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

def train_model(model, criterion, optimizer, scheduler, num_epochs=3):
    since = time.time()
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'Train':
                scheduler.step()
                model.train(True)      # Set model to training mode
            else:
                model.train(False)     # Set model to evaluation mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for data in data_loaders[phase]:
                # get the inputs 
                inputs, labels = data
                
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                
                # backward + optimize if only in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

                
            # save the model
            torch.save(model.state_dict(), 'Landmark_Resnet18_epoch' + str(epoch) +'.pth.tar')    
                 
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                
        print()
        
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Validation accuracy {:.4f}'.format(best_acc))
    
    # load the best model weights
    model.load_state_dict(best_model_wts)

    return model
    
    

landmark__model = models.resnet18(pretrained=True)
num_ftrs = landmark__model.fc.in_features
landmark__model.fc = nn.Linear(num_ftrs, 14944)

if use_gpu:
    landmark__model = landmark__model.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all the parameters have been optimized
optimizer_ft = optim.Adam(landmark__model.parameters(), lr=0.01)

# Decay learning rate by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

landmark__model = train_model(landmark__model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=120)

