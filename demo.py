from math import isnan, nan
import matplotlib
from matplotlib import pyplot as plt
import torch
import cv2
import numpy as np
import time
# Work in the parent directory
import os
import model
import torch
import os
import cv2
import numpy as np
import re
from pathlib import Path

from glob import glob
from itertools import repeat
from tqdm import tqdm, trange
from PIL import Image
from torchvision import transforms
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
torch.backends.cudnn.benchmark = True

def crop_input_data(left_img, crop_w, crop_h):
    crop_idx = [0, 0]

    # w, h = left_img.size
    x1 = crop_idx[0]
    y1 = crop_idx[1]

    # crop
    left_img_cropped = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
    left_img_cropped = np.array(left_img_cropped)
    
    return left_img_cropped

if __name__ == '__main__':
    filepath = '' # your dataset path
    path_limg = sorted( glob(os.path.join(filepath, os.path.join('left/*.png'))))
    
    model_path = '' # your model path

    crop_w, crop_h = 1280, 704
    
    start_epoch = 30
    end_epoch = 30
    
    opt_time = False
    all_time_opt = False
    
    output_path = 'demo_out'
    
    if not os.path.exists(output_path +'/'):
        os.makedirs(output_path +'/')

    

    # system setting
    f = 3515.99
    baseline = 0.09026 # unit m
    x_0 = torch.arange(0, int(crop_w/2.0)).unsqueeze(0)
    offset = 288 / 2.0
    
    # model
    model = torch.load(model_path, map_location="cuda")
    model.eval()
    
    start = 0
    end = 10
    for i in range(start, end+1):
        limg_path = path_limg[i]

        limg_full = Image.open(limg_path).convert('L')
        left = crop_input_data(limg_full, crop_w, crop_h)
        left_tmp = left
        max_disp = 288
        left = left.astype(np.float32) / 255.0
        left = torch.tensor(left, device="cuda").unsqueeze(0).unsqueeze(0)
        
        # infer    
        x_p = model(left)

        x_p = x_p[0, 0, :, :].cpu()
        x_p = x_p * (float(crop_w/2.0) + offset)
        x_p = x_p - x_0 - offset
        x_p = -(x_p.detach().numpy())

        disp = cv2.resize(x_p, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST) * 2
        depth = f * baseline / disp
            
        cv2.imwrite(output_path + output_path +'/' + str(i+1) + '_depth.exr', depth)