import numpy as np
import os
from glob import glob
import sys
import math
from random import randint
import random
from functools import partial

from utils import *
from Unet_util import *
from uNet import *
import time
import datetime


def imageblur(cimg, sampling=False):
    w, h, _ = cimg.shape
    blur = cv2.blur(cimg, (5, 5))
    for i in xrange(30):
        randx = randint(0,205)
        randy = randint(0,205)
        cimg[randx:randx+50, randy:randy+50] = 255
    hint =  cv2.blur(cimg,(100,100)) # background

    sample_num = 20
    threshold = 1
    for i in range(sample_num):
        x = randint(0,w-5)
        y = randint(0,h-5)
        """
        r = 12
        hint[x:x+r, y:y+r] = cimg[x:x+r, y:y+r]
        
        """
        # grow along the diagonal
        prev_mean = blur[x, y]
        r = 1
        while True:
            mean = np.mean(blur[x:x+r, y:y+r], axis = (0,1))
            minus = np.abs(mean - prev_mean)
            if minus[0] > threshold or minus[1] > threshold or minus[2] > threshold:
                break
            prev_mean = mean
            if x+r >= w or y+r >= h:
                break
            r += 1
        hint[x:x+r, y:y+r] = blur[x:x+r, y:y+r]
    return hint

def sample_colorblock(cimg, sampling=False):
    w, h, _ = cimg.shape
    cimg = cv2.blur(cimg, (5, 5))
    sample_num = 30
    threshold = 1
    hint = 255*np.ones_like(cimg)
    for i in range(sample_num):
        x = randint(0,w-5)
        y = randint(0,h-5)
        # grow along the diagonal
        prev_mean = cimg[x, y]
        r = 1
        while True:
            
            mean = np.mean(cimg[x:x+r, y:y+r], axis = (0,1))
            minus = np.abs(mean - prev_mean)
            if minus[0] > threshold or minus[1] > threshold or minus[2] > threshold:
                break
            prev_mean = mean
            if x+r >= w or y+r >= h:
                break
            r += 1
        print r
        hint[x:x+r, y:y+r] = cimg[x:x+r, y:y+r]
    return hint

val_data = glob(os.path.join("val","*.jpg"))
val = np.array([get_image(sample_file) for sample_file in val_data[0:4]])
val_normalized = val/255.0

val_edge = np.array([edge_detection(ba) for ba in val]) / 255.0
val_edge = np.expand_dims(val_edge, 3)

val_colors = np.array([imageblur(ba) for ba in val]) / 255.0
# val_colors = np.array([cv2.threshold(ba,255,255,cv2.THRESH_BINARY) for ba in val]) / 255.0

ims("1000Results/val.jpg",merge_color(val_normalized, [1, 4]))
ims("1000Results/val_line.jpg",merge(val_edge, [1, 4]))
ims("1000Results/val_colors.jpg",merge_color(val_colors, [1, 4]))