#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import glob 

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import pdb 
import shutil 
from path import Path 

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return im, inds 

    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
            (255, 255, 0), 2)

        cv2.putText(im, str(class_name)+str(score), (int(bbox[0]), int(bbox[1]-2)), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))

        return im, inds         


def demo(sess, net, image_name, newdir):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)

    im_file = image_name 
    im = cv2.imread(im_file)

    name, ext = os.path.splitext(im_file)
    text_file = os.path.basename(name)

    filename = [os.path.basename(im_file)]

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8 ## Only for pedestrian 
    NMS_THRESH = 0.3

    dets_out = []
    n = filename[0]

    # for cls_ind, cls in enumerate(CLASSES[1:]):
        # cls_ind += 1 # because we skipped background

    cls_ind = 15 
    cls = 'person'
    
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    im, inds = vis_detections(im, cls, dets, thresh=CONF_THRESH)

    if len(inds) > 0:
        dets_out.extend(dets[inds, :])

    dets_out = np.array(dets_out)

    dirc = newdir + text_file + '.txt'
    np.savetxt(dirc, dets_out)

    cv2.imwrite(newdir + n, im)
    return dets_out


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')

    ## name of bag as argument 
    # parser.add_argument('--fname', dest='name', help='Name of bag file')

    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 21,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #             '001763.jpg', '004545.jpg']

    root = Path('/home/aashi/tf-faster-rcnn/data/cmu_data/left_imgs_dec')
    scenes = root.dirs()

    for scene in scenes:

        # name = args.name 

        # im_names = glob.glob('/home/aashi/tf-faster-rcnn/data/cmu_data/' + str(name) + '/*.png')
        im_names = glob.glob(scene + '/*.png')

        name = os.path.split(scene)[-1]

        newdir = '/home/aashi/tf-faster-rcnn/data/cmu_data/det_left_dec/det_' + str(name) + '/'

        if os.path.exists(newdir):
            shutil.rmtree(newdir)

        os.makedirs(newdir)

        for im_name in im_names:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Demo for data/demo/{}'.format(im_name))
            demo(sess, net, im_name, newdir)

    root = Path('right_imgs') # Image Folder 
    scenes = root.dirs()

    for scene in scenes:

        im_names = glob.glob(scene + '/*.png')

        name = os.path.split(scene)[-1]

        newdir = 'det_' + str(name) + '/'

        if os.path.exists(newdir):
            shutil.rmtree(newdir)

        os.makedirs(newdir)

        for im_name in im_names:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Demo for data/demo/{}'.format(im_name))
            demo(sess, net, im_name, newdir)

    # plt.show()
