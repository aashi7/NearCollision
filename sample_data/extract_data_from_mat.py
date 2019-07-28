import scipy.io
import os 
import matplotlib.pyplot as plt
import shutil 
import cv2 

mat = scipy.io.loadmat('sample.mat')

left_imgs = mat['left_imgs']

newdir = 'sample_left_imgs/'

if os.path.exists(newdir):
	shutil.rmtree(newdir)

os.makedirs(newdir)

for i in range(left_imgs.shape[1]):
	cv2.imwrite(newdir + str(i+1) + '.png', left_imgs[0][i])

