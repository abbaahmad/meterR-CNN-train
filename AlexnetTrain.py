import MeterDataloader as Dataloader
from torch.utils import data
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim

params = {
	'batch_size':64,
	'shuffle' : True,
	'num_workers':1 }
max_epoch=7

model = models.alexnet(pretrained=True)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features,3)
#print("Alexnet is: {}".format(model))
#exit()
criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

dataset = Dataloader.MeterDataset('data/images')
train_loader = data.DataLoader(dataset,**params)
#exit()
for epoch in range(max_epoch):
	print("Epoch {}/{}".format(epoch,max_epoch-1))
	print('-'*10)
	
	running_loss = 0.0
	for image, label in train_loader:
		#print("Label:{}\n".format(label))
		#exit()
		optimiser.zero_grad()
		#label_t = torch.Tensor(4)
		#label_t=torch.Tensor(data=label)
		#label_t = label_t1.view(4,)
		output = model(image)
		#output = output.view(4,1)
		#output = nn.functional.log_softmax(output,dim=1)
		#print("Input shape is {}".format(image.size()))
		#print("Input is {}".format(image))
		#print("Output shape is {}".format(output.size()))
		#print("label shape is {}".format(label.size()))
		print("Label:{}\n".format(label))
		print("Output is {}\n".format(output))
		#exit()
		loss = criterion(output,label)#torch.max(label_t, -1)[1])
		print("Loss is {}".format(loss.item()))
		loss.backward()
		print("Backward loss and now optimiser")
		optimiser.step()
		
		running_loss += loss.item()
		print("Loss: {:.4f}".format(running_loss))
		#exit()
