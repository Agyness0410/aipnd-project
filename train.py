# coding=utf-8
import time
import copy
import numpy as np
import pandas as pd
import os
import torch
import json
from PIL import Image
import torchvision
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch import nn,optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='flowers', help='path to the folder of flower images')
parser.add_argument('--save_dir', type=str, default='./savedmodels/', help='The model save dir')
parser.add_argument('--arch', type=str, default='densenet', help='The CNN model architecture to use')
parser.add_argument('--learning_rate', type=float, default=0.01, help='The learning rate')
parser.add_argument('--hidden_units', type=int, default=512, help='The hidden layer')
parser.add_argument('--epochs', type=int, default=5, help='epochs train loop')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size for one train')
parser.add_argument('--gpu', type=str, default='cuda:0', help='The device for use train')
parser = parser.parse_args()

data_dir = parser.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30)
                            , transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip()
                            ,transforms.ToTensor()
                            ,  transforms.Normalize([0.485, 0.456, 0.406]
                            , [0.229, 0.224, 0.225])]),
                   'valid': transforms.Compose([transforms.Resize(256)
                   ,transforms.CenterCrop(224), transforms.ToTensor()
                   , transforms.Normalize([0.485, 0.456, 0.406]
                   ,[0.229, 0.224, 0.225])])}


image_datasets = {x: ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=parser.batch_size, shuffle=True, num_workers=0)  for x in ['train', 'valid']}
class_names = image_datasets['train'].classes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model_name = parser.arch
if model_name == 'densenet':
    model = models.densenet161(pretrained=True)
    num_in_features = 2208
elif model_name == 'vgg':
    model = models.vgg19(pretrained=True)
    num_in_features = 25088
else:
    print("Unknown model, please choose 'densenet' or 'vgg'")

for param in model.parameters():
    param.requires_grad = False


def build_classifier(num_in_features, hidden_layers, num_out_features):
    classifier = nn.Sequential()
    if hidden_layers == None:
        classifier.add_module('fc0', nn.Linear(num_in_features, 102))
    else:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        classifier.add_module('fc0', nn.Linear(num_in_features, hidden_layers[0]))
        classifier.add_module('relu0', nn.ReLU())
        classifier.add_module('drop0', nn.Dropout(.6))
        classifier.add_module('relu1', nn.ReLU())
        classifier.add_module('drop1', nn.Dropout(.5))
        for i, (h1, h2) in enumerate(layer_sizes):
            classifier.add_module('fc' + str(i + 1), nn.Linear(h1, h2))
            classifier.add_module('relu' + str(i + 1), nn.ReLU())
            classifier.add_module('drop' + str(i + 1), nn.Dropout(.5))
        classifier.add_module('output', nn.Linear(hidden_layers[-1], num_out_features))

    return classifier

hidden_layers = None#[4096, 1024, 256][512, 256, 128]

classifier = build_classifier(num_in_features, hidden_layers, 102)


 # Only train the classifier parameters, feature parameters are frozen
if model_name == 'densenet':
    model.classifier = classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters())
    # optim.Adam(model.parameters(), lr=parser.learning_rate, momentum=0.9) # Adadelta
    # optimizer_conv = optim.SGD(model.parameters(), lr=parser.learning_rate, weight_decay=0.001, momentum=0.9) #weight
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4)
elif model_name == 'vgg':
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=parser.learning_rate)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
else:
    pass

device = torch.device(parser.gpu if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, sched, num_epochs=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # sched.step()
                        loss.backward()

                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


model.to(device)
model = train_model(model, criterion, optimizer, sched, parser.epochs)

model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'input_size': 2208,
              'output_size': 102,
              'epochs': parser.epochs,
              'batch_size': parser.batch_size,
              'model': models.densenet161(pretrained=True),
              'classifier': classifier,
              'scheduler': sched,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx
              }

torch.save(checkpoint, parser.save_dir + 'checkpoint_model_ic_py.pth')

model.eval()

accuracy = 0

for inputs, labels in dataloaders['valid']:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)

    # Class with the highest probability is our predicted class
    equality = (labels.data == outputs.max(1)[1])

    # Accuracy is number of correct predictions divided by all predictions
    accuracy += equality.type_as(torch.FloatTensor()).mean()

print("Test accuracy: {:.3f}".format(accuracy / len(dataloaders['valid'])))
