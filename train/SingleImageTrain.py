
# coding: utf-8

# In[1]:


import torch
import torchvision.models as models
import h5py 
from logger import Logger
from torchvision.transforms import transforms 
import torch.utils.data as data
import numpy as np 
import pdb
import matplotlib.pyplot as plt
import torch.nn as nn 
import torch.optim as optim 
from torch.autograd import Variable
import shutil
import os 
import random
import torch.nn.functional as F
import math 


# In[2]:


class FrameDataset(data.Dataset):
    
    def __init__(self, f, transform=None, test = False):
        self.f = f 
        self.transform = transform 
        self.test = test
        
    def __getitem__(self, index):

        rgb = np.array(self.f["rgb"][index])
        label = np.array((self.f["labels"][index] - self.f["Mean"])/self.f["Variance"])
        
        t_rgb = torch.zeros(3, 224, 224)
        
        prob = random.uniform(0, 1)
        prob2 = random.uniform(0, 1)

        if self.transform is not None:
            if (prob > 0.5 and not self.test):
                flip_transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(1.0)])
                rgb[:,:,:] = flip_transform(rgb[:,:,:])
            if (prob2 > 0.5 and not self.test):
                color_jitter_transform = transforms.Compose([transforms.ToPILImage() ,transforms.ColorJitter(brightness = 0.5, contrast = 0.5, saturation = 0.5, hue = 0.2)])
                rgb[:,:,:] = color_jitter_transform(rgb[:,:,:])

            t_rgb[:,:,:] = self.transform(rgb[:,:,:])                
        
        return t_rgb, label
    
    def __len__(self):
        return len(self.f["rgb"])


# In[3]:


def load_model_weights(MODEL_PATH):
    checkpoint_dict = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint_dict)
    
def save_model_weights(epoch_num):
    model_file = '/mnt/hdd1/aashi/SingleImage6s_' + str(epoch_num).zfill(3)
    torch.save(model.state_dict(), model_file)


# In[4]:


model = models.vgg16(pretrained=True)
num_final_in = model.classifier[-1].in_features
NUM_CLASSES = 20
model.classifier[-1] = nn.Linear(num_final_in, NUM_CLASSES)

model_path = '/home/aashi/the_conclusion/model_files/' + 'vgg_on_voc' + str(800)
load_model_weights(model_path)

model.classifier[-1] = nn.Linear(num_final_in, 1) ## Regressed output
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 2048), nn.ReLU(), nn.Linear(2048, 1)]) # Add our layer with 4 outputs
model.classifier = nn.Sequential(*features) # Replace the model classifier

# In[5]:


if os.path.exists('SingleImage6s'):
    shutil.rmtree('SingleImage6s')
logger = Logger('SingleImage6s', name='performance_curves')


# In[6]:


model = model.cuda()

hfp_test = h5py.File('/mnt/hdd1/aashi/cmu_data/SingleImageTest.h5', 'r')
hfp_train = h5py.File('/mnt/hdd1/aashi/cmu_data/SingleImageTrain.h5', 'r')
mean = hfp_train["Mean"][()]
var = hfp_train["Variance"][()]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_loader = data.DataLoader(FrameDataset(f = hfp_test, transform = transforms.Compose([transforms.ToTensor(), normalize]), test = True), 
                               batch_size=1)
batch_size = 28
train_loader = data.DataLoader(FrameDataset(f = hfp_train, transform = transforms.Compose([transforms.ToTensor(), normalize]),test = False),
                              batch_size=batch_size, shuffle=True)
optimizer = optim.SGD(model.parameters(),0.0005)

## Test data: 949; Train data: 12146; Mean: 2.909sec; Variance: 2.2669sec

iterations = 0
epochs = 50 
criterion = nn.MSELoss()

for e in range(epochs):
    
    print(e)
    save_model_weights(e)
    model.eval()
    err = 0.0
    for iter,(rgb, label) in enumerate(test_loader, 0):
        rgb = rgb.float().cuda()
        label = Variable(label.float().cuda())
        label = label.unsqueeze(-1)
        outputs = model(rgb)
        err += abs(outputs[0].data.cpu().numpy() - label[0].data.cpu().numpy())
    logger.scalar_summary('Test Error', err/len(test_loader), e)
    model.train()
    
    for iter, (rgb, label) in enumerate(train_loader, 0):

        rgb = Variable(rgb.float().cuda())
        label = Variable(label.float().cuda())
        optimizer.zero_grad()
        label = label.unsqueeze(-1)
        outputs = model(rgb)
        loss = criterion(outputs, label) 
        loss.backward()
        optimizer.step()
        iterations += 1
        logger.scalar_summary('training_loss', loss.data.cpu().numpy(), iterations)


# In[ ]:


hfp_test.close()
hfp_train.close()

