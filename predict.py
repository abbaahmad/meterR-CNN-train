import os
import argparse
import numpy as np
from PIL import Image
import MeterDataloader as Dataloader

import torch
from torchvision import models, transforms
import torch.nn as nn
from torch.utils import data

ap = argparse.ArgumentParser()
ap.add_argument("-m","--model",required=True,
				help="path to model")
ap.add_argument("-i","--image", required=True,
				help="path to test image")
ap.add_argument("-r","--root",default="data/images",
				help="path to root dir of images")				
args=vars(ap.parse_args())


classes = ('display','meter','not_meter')

img_array = Image.open(args['image']).convert('RGB')
preprocess = transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
		])
img = preprocess(img_array)
image = img.reshape(-1,3,224,224)


checkpoint = torch.load(args['model'])	
model = models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features,3)			
model.load_state_dict(checkpoint['model_state_dict']) 

#print(model)

model.eval()

output = model(image)

_,preds_tensor= torch.max(output,1)
preds = np.squeeze(preds_tensor.numpy())
result = classes[preds]
print("Prediction is {} of class {}".format(preds,result))
