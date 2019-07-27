
# coding: utf-8

# In[22]:


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


# In[23]:


class FrameDataset(data.Dataset):
    
    def __init__(self, f, transform=None, test = False):
        self.f = f 
        self.transform = transform 
        self.test = test
        
    def __getitem__(self, index):
        if (not self.test):
            rgb = np.array(self.f["rgb"][index])
            #label = np.array(self.f["MeanSubtractedLabels"][index])
            label = np.array((self.f["labels"][index] - self.f["Mean"]))
        else:
            rgb = np.array(self.f["rgb"][index])
            label = np.array((self.f["labels"][index] - self.f["Mean"])) ## same mean through train, test 
        
        t_rgb = torch.zeros(rgb.shape[0], 3, 224, 224)
        
        prob = random.uniform(0, 1)
        prob2 = random.uniform(0, 1)

        if self.transform is not None:
            for i in range(rgb.shape[0]):
                if (prob > 0.5 and not self.test):
                    flip_transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(1.0)])
                    rgb[i,:,:,:] = flip_transform(rgb[i,:,:,:])
                t_rgb[i,:,:,:] = self.transform(rgb[i,:,:,:])

                
        return t_rgb, label
    
    def __len__(self):
        return len(self.f["rgb"])


# In[24]:


def load_vgg_voc_weights(MODEL_PATH):
    checkpoint_dict = torch.load(MODEL_PATH)
    vgg_model.load_state_dict(checkpoint_dict)

vgg_model = models.vgg16(pretrained=True)
num_final_in = vgg_model.classifier[-1].in_features
NUM_CLASSES = 20 ## in VOC
vgg_model.classifier[-1] = nn.Linear(num_final_in, NUM_CLASSES)
model_path = '/home/aashi/the_conclusion/model_files/' + 'vgg_on_voc' + str(800)
load_vgg_voc_weights(model_path)


# In[25]:


class VGGNet(nn.Module):
    
    def __init__(self):
        super(VGGNet, self).__init__()
        self.rgb_net = self.get_vgg_features()
        
        kernel_size = 3 
        padding = int((kernel_size - 1)/2)
        self.conv_layer = nn.Conv2d(512, 16, kernel_size, 1, padding, bias=True)
        #self.conv_bn = nn.BatchNorm2d(16)
        ## input_channels, output_channels, kernel_size, stride, padding, bias
        self.feature_size = 16*7*7*6
        #self.feature_size = 16*7*7*3 
        self.final_layer = nn.Sequential(
        nn.ReLU(),
        nn.Linear(self.feature_size, 2048),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(2048, 1)
        #nn.Sigmoid()
        #nn.Softmax()  ## If loss function uses Softmax  
        )
        
    def forward(self, rgb): ## sequence of four images - last index is latest 
        four_imgs = []
        for i in range(rgb.shape[1]):
            img_features = self.rgb_net(rgb[:,i,:,:,:])
            channels_reduced = self.conv_layer(img_features)
            img_features = channels_reduced.view((-1, 16*7*7))
            four_imgs.append(img_features)
        concat_output = torch.cat(four_imgs, dim = 1)
        out = self.final_layer(concat_output)
        return out
#         return concat_output
        
    def get_vgg_features(self):

        modules = list(vgg_model.children())[:-1]
        vgg16 = nn.Sequential(*modules)
        
        ## Uncommented this to let it fine-tune on my model 
        # for p in vgg16.parameters():
        #     p.requires_grad = False 
        
        return vgg16.type(torch.Tensor)


# In[26]:


model = VGGNet().cuda()


# In[27]:


# if os.path.exists('6Image6sCorrected'):
#     shutil.rmtree('6Image6sCorrected')
# logger = Logger('6Image6sCorrected', name='performance_curves')

if os.path.exists('6Image6sd2'):
    shutil.rmtree('6Image6sd2')
logger = Logger('6Image6sd2', name='performance_curves')

def save_model_weights(epoch_num):
    #model_file = '/mnt/hdd1/aashi/6Image6sCorrected_' + str(epoch_num).zfill(3)
    model_file = '/mnt/hdd1/aashi/6Image6sd2_' + str(epoch_num).zfill(3)
    torch.save(model.state_dict(), model_file)    


def load_model_weights(MODEL_PATH):
    checkpoint_dict = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint_dict)

# In[28]:


# hfp_test = h5py.File('/mnt/hdd1/aashi/cmu_data/6ImageTestCorrected.h5', 'r')
# hfp_train = h5py.File('/mnt/hdd1/aashi/cmu_data/6ImageTrainCorrected.h5', 'r')

hfp_test = h5py.File('/mnt/hdd1/aashi/cmu_data/6ImageTestd2.h5', 'r')
hfp_train = h5py.File('/mnt/hdd1/aashi/cmu_data/6ImageTraind2.h5', 'r')

mean = hfp_train["Mean"][()]
var = hfp_train["Variance"][()]
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_loader = data.DataLoader(FrameDataset(f = hfp_test, transform = transforms.Compose([transforms.ToTensor(), normalize]), test = True), 
                               batch_size=1)
#bacth_size= 12, lr = 0.001 
#batch_size = 20
batch_size = 24
print(batch_size)
train_loader = data.DataLoader(FrameDataset(f = hfp_train, transform = transforms.Compose([transforms.ToTensor(), normalize]),test = False),
                              batch_size=batch_size, shuffle=True)
#optimizer = optim.SGD(model.parameters(),0.0005)
optimizer = optim.SGD(model.parameters(),0.001)

# In[29]:


print(len(test_loader))

# In[30]:


# len(train_loader)
# MODEL_PATH = '/mnt/hdd1/aashi/6Image6s_049'
# load_model_weights(MODEL_PATH)
# print('Previous model loaded')
# In[ ]:

## 6 frames
## Test data: 1038; Train data: 12620; Mean: 2.98 sec; Variance: 2.299 sec

iterations = 0
epochs = 50 
criterion = nn.MSELoss()

for e in range(0,epochs):
    
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
        logger.scalar_summary('training_loss v2', loss.data.cpu().numpy(), iterations)

    
    save_model_weights(e)
    model.eval()
    err = []
    for iter,(rgb, label) in enumerate(test_loader, 0):
        rgb = rgb.float().cuda()
        label = Variable(label.float().cuda())
        label = label.unsqueeze(-1)
        outputs = model(rgb)
        err.append(abs(outputs[0].data.cpu().numpy() - label[0].data.cpu().numpy()))
    logger.scalar_summary('Test Error v2', np.mean(err), e)
    print(e, np.mean(err), np.std(err))

