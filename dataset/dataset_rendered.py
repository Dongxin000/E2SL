import torch.utils.data as data
import numpy as np
import cv2
import os
import re
from common.common import *
from pathlib import Path
from PIL import Image
import random

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import sys 
sys.path.append("..")

class DatasetSL(data.Dataset):
    def __init__(self, path_lists, train_opt):
        self.left_filenames = path_lists[0]
        self.right_filenames = path_lists[1]
        self.disp_filenames = path_lists[2]
        self.msk_filenames = path_lists[3]
        self.rawprj_filenames = path_lists[4]
        self.rawbg_filenames = path_lists[5]
        self.e2vid_filenames = path_lists[6]
        
        self.f = 3515.99
        self.baseline = 0.09026 # unit m

        self.train_opt = train_opt
    def __len__(self):
        if len(self.left_filenames) == 0:
            return len(self.rawbg_filenames)
        else:
            return len(self.left_filenames)

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data
    

    def process_black_img(self, binary_img, density=0.01, seed=None):
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        img_mask = rng.random(binary_img.shape) < density
        
        binary_img = binary_img.copy()
        binary_img[(binary_img == 0) & img_mask] = 255
        
        return binary_img

    def add_hot_pixel_noise(self, im):
        H = 704
        W = 1280

        hot_pixel_fraction = np.random.uniform(0.0, 0.0001)

        hot_pixel_mask = np.random.rand(H, W) < hot_pixel_fraction
        noisy_img = im.copy()
        noisy_img[hot_pixel_mask != 0] = 255


        return noisy_img

    
    def log_binary_add_noise(self, imPrj, imBg, t_log):
        if imPrj.ndim == 3:
            prj = (imPrj[:, :, 0] + imPrj[:, :, 1] + imPrj[:, :, 2])/3.0
            bg = (imBg[:, :, 0] + imBg[:, :, 1] + imBg[:, :, 2])/3.0
        else:
            prj = imPrj
            bg = imBg
        
        t_np = np.full(prj.shape, t_log, dtype=np.float32)

        sigma = np.random.uniform(0.005, 0.015)
        delta_t = np.random.normal(loc=0.0, scale=sigma, size=prj.shape)

        t_np = t_np + delta_t
        t_np = np.clip(t_np, 0.1, 0.4)

        prj_log = np.log2(prj)
        bg_log = np.log2(bg)

        diff_log = prj_log - bg_log

        base_noise = np.random.normal(loc=0.0, scale=0.1, size=prj.shape)
        diff_log = diff_log + base_noise

        binary_msk = np.where(diff_log > t_np, 255, 0).astype(np.uint8)

        binary_msk = self.add_hot_pixel_noise(binary_msk)

        return binary_msk

    
    def inverse_gamma_on_unit_interval(self, x: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        y = x.copy()
        mask = (x >= 0.0) & (x <= 1.0)
        y[mask] = np.power(x[mask], gamma).astype(np.float32)

        return y

    def __getitem__(self, idx):    
        if self.train_opt:
            rawprj = cv2.imread(os.path.join(self.rawprj_filenames[idx]), cv2.IMREAD_UNCHANGED)
            rawbg = cv2.imread(os.path.join(self.rawbg_filenames[idx]), cv2.IMREAD_UNCHANGED)
        else:
            left_img = self.load_image(os.path.join(self.left_filenames[idx]))
        
        disparity = self.load_disp(os.path.join(self.disp_filenames[idx]))
        msk = self.load_image(os.path.join(self.msk_filenames[idx]))


        # init
        # crop_idx = [316, 82, 1601, 797]
        crop_idx = [316, 82]
        crop_w, crop_h = 1280, 704

        x1 = crop_idx[0]
        y1 = crop_idx[1]
        
        if self.train_opt:
            vertical_jitter = 2
            v_offset = np.random.randint(-vertical_jitter, vertical_jitter)

            rawprj_cropped = rawprj[y1 + v_offset:y1 + v_offset + crop_h, x1:x1 + crop_w]
            rawbg_cropped = rawbg[y1 + v_offset:y1 + v_offset + crop_h, x1:x1 + crop_w]

            t_log = random.uniform(a=0.1, b=0.4)
            left_img = np.array(self.log_binary_add_noise(rawprj_cropped, rawbg_cropped, t_log))

        else:
            left_img = np.array(left_img)
            v_offset = 0
    
        left_img = left_img.astype(np.float32) * (1.0 / 255.0)
        
        # crop
        if self.train_opt:
            left_img_cropped = left_img
        else:
            left_img_cropped = left_img[y1 + v_offset:y1 + v_offset + crop_h, x1:x1 + crop_w]

        msk_cropped = msk.crop((x1, y1 + v_offset, x1 + crop_w, y1 + v_offset + crop_h))
        disparity_cropped = disparity[y1 + v_offset:y1 + v_offset + crop_h, x1:x1 + crop_w]

        if len(np.unique(left_img_cropped))==1:
            if len(np.unique(msk_cropped))==1:
                left_img_cropped = self.process_black_img(left_img_cropped, density=0.01)
        msk_cropped = np.array(msk_cropped).astype(np.float32)
        depth_cropped = (self.f * self.baseline) / disparity_cropped

        
        msk_cropped[depth_cropped > 20] = 0
        msk_cropped[msk_cropped == 255] = 1

        
        # half resolution
        depth_cropped = downsampleDepth(depth_cropped)
        
        offset = 288 / 2.0
        
        # tranfer disp to x_p (in projector/right camera) 
        disparity_cropped = ((self.f/2.0) * self.baseline) / depth_cropped
        x_p = -disparity_cropped + np.expand_dims(np.arange(0, int(crop_w/2)), 0).astype(np.float32) + offset

        
        max_disp = 288.0 / 2.0
        x_p = (x_p) * (1.0 / (float(crop_w/2.0) + offset))  # normalize between 0 and 1 (above and below are not impossible

        # downsample the mask. (prioritize invalid pixel!!!)
        msk_cropped[msk_cropped == 0] = 2
        msk_cropped = downsampleDepth(msk_cropped)
        msk_cropped[msk_cropped == 2] = 0
        msk_cropped[disparity_cropped > max_disp] = 0
        
        left_img_cropped = np.expand_dims(left_img_cropped, 0)
        msk_cropped = np.expand_dims(msk_cropped, 0)
        x_p = np.expand_dims(x_p, 0)
        
        # Create Edge map by executing sobel on depth
        #  threshold
        depth_1 = 1.0 / depth_cropped
        depth_1[np.isnan(depth_1)] = 0
        grad_x = cv2.Sobel(depth_1, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_1, cv2.CV_32F, 0, 1, ksize=3)
        edge_threshold = 0.1  # a 10 centimeter threshold!!!!
        edges = (grad_x * grad_x + grad_y * grad_y) > edge_threshold * edge_threshold
        edges = edges.astype(np.float32)
        #  dilate
        edges = dilatation(edges, 10)
        edges = np.expand_dims(edges, 0)
        
        return left_img_cropped, x_p, msk_cropped, edges
    