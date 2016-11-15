#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
count_im = 1
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def print_results(im, class_name, dets, thresh):
    
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    
    for i in inds:
    #if class_name == "car":
	#print(dets[inds])
        score = dets[i, -1]
	#dets[inds[0][:,2:4]] += dets[inds[0][:,0:2]]
	#dets[inds][:,2:4] = dets[inds][:,2:4] + dets[inds][:,0:2]
	dets[:, 2] = dets[:,2] + dets[:,0]
	dets[:, 3] = dets[:,3] + dets[:,1]
	#print(dets[inds])
	
    #return dets[inds]

    print '{:s} {:.3f}'.format(class_name,score)


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    global inds, dets
    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    #im = cv2.imread(image_name)
    im = image_name
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for ' '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.6
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
	if cls == "car":
	    inds = np.where(dets[:, -1] >= 0.6)[0]
	    dets[:, 2] = dets[:,2] + dets[:,0]
            dets[:, 3] = dets[:,3] + dets[:,1]
	    #print(dets[inds])
	    to_ret = dets[inds]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
	print_results(im, cls, dets, thresh=CONF_THRESH)
    #print "to return values: {}".format(to_ret)
    return to_ret

#if __name__ == '__main__':
def initialize():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    #args = parse_args()

    prototxt = '/home/ls/FRCN_ROOT/RCNN_model/Pretrained_zf/Pretrained_zf.pt' 
    caffemodel = '/home/ls/FRCN_ROOT/RCNN_model/Pretrained_zf/Pretrained_zf.caffemodel'
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    #if args.cpu_mode:
        #caffe.set_mode_cpu()
    #else:
    caffe.set_mode_gpu()
    #caffe.set_device(0)
    cfg.GPU_ID = 0
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    global net
    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    print 'warmup successful \n'


class cafferun():
    global count_im
    def __init__(self):
        initialize()
   
    #im_names = ['detectr.jpg']
    #for im_name in im_names:
    def run(self,im_name):
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        #print 'Demo for data/demo/{}'.format(im_name)
        det_send = demo(net, im_name)
        name = "%count_im.jpg" % count_im
	plt.savefig(name, format='jpg')
	return det_send
