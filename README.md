# Forecasting Time-to-Collision from Monocular Video: Feasibility, Dataset and Challenges  

<img align="center" src="https://github.com/aashi7/NearCollision/blob/master/demo.gif">

This repository contains the official pytorch implementation for "[Forecasting Time-to-Collision from Monocular Video: Feasibility, Dataset and Challenges](https://arxiv.org/pdf/1903.09102.pdf)". Please also check out our project page [here](https://aashi7.github.io/NearCollision.html). If you find this code useful, please cite our paper:

```
@article{Manglik2019, 
  archivePrefix = {arXiv}, 
  arxivId = {1903.09102}, 
  author = {Manglik, Aashi and Weng, Xinshuo and Ohn-bar, Eshed and Kitani, Kris}, 
  eprint = {1903.09102}, 
  journal = {arXiv:1903.09102}, 
  title = {{Forecasting Time-to-Collision from Monocular Video: Feasibility, Dataset and Challenges}}, 
  url = {https://arxiv.org/pdf/1903.09102.pdf}, 
  year = {2019} 
}
```

### Dataset 

Here is [link](https://drive.google.com/drive/u/1/folders/1tAywlmXA3iDJtggUIP3FFh0RoBXtH1Qu
) to our dataset. The data is stored ```.mat``` format.  

|S.No. | Folder  | Recordings       |         
|-| ------------------ | -------      |  
|1|  mats_nov         | Left images, Right images, 3D point cloud, Calibration       |   
|2|  mats_dec       | Left images, Right images, 3D point cloud, Calibration     |   
|3|  mat_stereo_camera  | Left images, Right images, Depth Maps from Stereo Camera|  


### Requirements
- Python 2.7
- [PyTorch](https://pytorch.org/) 

### Installation

#### Setup virtualenv


### Demo
1. Download the trained model:


2. Run the demo:


### Training
Please see 
