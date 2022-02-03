# coding=utf-8
import time
import copy
import numpy as np
import pandas as pd
import os
import torch
from PIL import Image
import torchvision
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default='flowers/valid/102/image_08038.jpg',
                    help='path to the folder of flower images')
parser.add_argument('--checkpoint', type=str, default='checkpoint_model_ic_py.pth',
                    help='path to the folder of flower images')
parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping categories real names')
parser.add_argument('----top_k', type=int, default=5, help='mapping categories real names')
parser.add_argument('--gpu', type=str, default='cuda:0', help='The device for use predict')
parser = parser.parse_args()

batch_size = 8
data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30)
                                                   , transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip()
                                                   , transforms.ToTensor()
                                                   , transforms.Normalize([0.485, 0.456, 0.406]
                                                                          , [0.229, 0.224, 0.225])]),
                   'valid': transforms.Compose([transforms.Resize(256)
                                                   , transforms.CenterCrop(224), transforms.ToTensor()
                                                   , transforms.Normalize([0.485, 0.456, 0.406]
                                                                          , [0.229, 0.224, 0.225])])}

image_datasets = {x: ImageFolder(os.path.join('flowers', x), data_transforms[x]) for x in ['train', 'valid']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

if parser.gpu == 'cuda:0':
    device = parser.gpu
else:
    device = 'cpu'

import json

with open(parser.category_names, 'r') as f:
    cat_to_name = json.load(f)



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']

    for param in model.parameters():
        param.requires_grad = False

    return model, checkpoint['class_to_idx']


model, class_to_idx = load_checkpoint('savedmodels/' + parser.checkpoint)


def process_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    return image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, topk=5):
    img = Image.open(image_path)
    img = process_image(img)

    # Convert 2D image to 1D vector
    img = np.expand_dims(img, 0)

    img = torch.from_numpy(img)

    model.eval()
    inputs = Variable(img).to(device)
    logits = model.forward(inputs)

    ps = F.softmax(logits, dim=1)
    topk = ps.cpu().topk(topk)

    return (e.data.numpy().squeeze().tolist() for e in topk)




img_path = parser.img_dir
probs, classes = predict(img_path, model.to(device))
print(probs)
print(classes)

if parser.category_names is not None:
    flower_names = [cat_to_name[class_names[e]] for e in classes]
    print(probs)
    print(flower_names)

if parser.top_k != None:
    probs, classes = predict(img_path, model.to(device), topk=parser.top_k)
    print(probs)
    print(classes)

