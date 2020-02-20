import MeterDataloader as Dataloader
from torch.utils import data
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim

params = {
	'batch_size':64,
	'shuffle' : True,
	'num_workers':0 }
max_epoch=7
print("[INFO] loading and customising alexnet model")
model = models.alexnet(pretrained=True)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features,3)
#print("Alexnet is: {}".format(model))
#exit()
criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

print("[INFO] loading data")
dataset = Dataloader.MeterDataset('data/images')
train_loader = data.DataLoader(dataset,**params)
#exit()
print("[INFO] starting training loop")
losses ={}
for epoch in range(max_epoch):
	print("Epoch {}/{}".format(epoch,max_epoch-1))
	print('-'*10)
	
	running_loss = 0.0
	for image, label in train_loader:
		
		
		optimiser.zero_grad()
		
		
		output = model(image)
		
		
		#print("Label:{}\n".format(label))
		#print("Output is {}\n".format(output))
		
		loss = criterion(output,label)#torch.max(label_t, -1)[1])
		#print("Loss is {}".format(loss.item()))
		loss.backward()
		#print("Backward loss and now optimiser")
		optimiser.step()
		
		running_loss += loss.item()
		losses[epoch] = loss.item()
		print("Loss: {:.4f}".format(running_loss))
		#exit()
		
print("[INFO] saving model to {}".format(os.getcwd()))
torch.save({
		'epoch':epoch,
		'model_state_dict':model.state_dict(),
		'optimiser_state_dict':optimiser.state_dict(),
		'losses':losses,
		'running_loss':running_loss
		}
		,os.path.join(os.getcwd(),'alexnet_model.pth'))