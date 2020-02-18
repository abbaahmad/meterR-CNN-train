#!/usr/bin/env
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import os
from PIL import Image
from imutils import paths
import random

class MeterDataset(torch.utils.data.Dataset):
	def __init__(self,root='data/images',transforms=None):
		self.root = root	#top-level data folder
		self.images = []
		self.labels_dict = {}
		self.classes = 0
		img_list=[]
		listOfList=[]
		self.abs_path = os.path.join(os.getcwd(),root)
		#print("[INFO] Root directory is {}".format(self.abs_path))

		for directory in os.listdir(self.abs_path):
			dir_path = os.path.join(self.abs_path,directory)
			img_list = list(sorted(os.listdir(dir_path)))
			self.labels_dict[self.classes] = directory
			listOfList.append([os.path.join(dir_path,x) for x in img_list])
			self.classes +=1 
		
		#Flatten listOfList
		self.images=[item for sublist in listOfList for item in sublist]
		random.seed(24)
		random.shuffle(self.images)
		
	def __getitem__(self,idx):
		#load image
		img_path = self.images[idx]
		label = img_path.split(os.path.sep)[-2]
		label = encoder(self.labels_dict,label)
		img_array = Image.open(img_path).convert('RGB')
		
		preprocess = transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
		])
		img = preprocess(img_array)
		
		return img, label
	
	def __len__(self):
		return len(self.images) 

def encoder(dictionary,value):
 	
	for k,v in dictionary.items():
		if v== value:
			return k			

def decoder(dictionary,key):
	
	for k,v in dictionary.items():
		return dictionary[k]		
