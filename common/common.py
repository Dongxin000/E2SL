import numpy as np
import scipy.signal
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

# load disp path
def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def downsample(image):
    image = image.reshape(int(image.shape[0]/2), 2, int(image.shape[1]/2), 2)
    image = np.mean(image, axis=3)
    image = np.mean(image, axis=1)
    return image

def downsampleDepth(d):
    d = d.reshape(int(d.shape[0]/2), 2, int(d.shape[1]/2), 2)
    d = np.min(d, axis=3)
    d = np.min(d, axis=1)
    return d

def downsampleDisp(d):
    d = d.reshape(int(d.shape[0]/2), 2, int(d.shape[1]/2), 2)
    d = np.max(d, axis=3)
    d = np.max(d, axis=1)
    return d

def dilatation(src, r):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    dilation_dst = cv2.dilate(src, element)
    return dilation_dst


# Local contrast normalization for numpy
def LCN_np(image, window_size=9):
    eps = 0.001
    width = image.shape[1]
    height = image.shape[0]
    im_padded = np.pad(image, int(window_size / 2), mode="edge")
    kernel = np.ones((window_size, 1), dtype=np.float32) * (1.0 / window_size)
    mean = scipy.signal.convolve2d(im_padded, kernel, mode="valid")
    kernel = np.ones((1, window_size), dtype=np.float32) * (1.0 / window_size)
    mean = scipy.signal.convolve2d(mean, kernel, mode="valid")
    cv2.imshow("mean_np", mean)
    sq = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    for i in range(window_size):
        for j in range(window_size):
            sq[:, :] += (im_padded[i:i+height, j:j+width] - mean) ** 2

    sigma = np.sqrt(sq/(window_size**2))
    cv2.imshow("std_np", sigma)
    I = (image - mean) / (sigma + eps)

    return I * 0.5 + 0.5

# Local contrast normalization for numpy
def LCN_tensors(x, window_size=9):
    eps = 0.001
    dtype = x.dtype
    device = x.device

    w = torch.ones((x.shape[1], x.shape[1], window_size, window_size), device=device, dtype=dtype) / \
        (window_size * window_size)
    mean_local = F.conv2d(input=x, weight=w, padding=window_size // 2)

    mean_square_local = F.conv2d(input=x ** 2, weight=w, padding=window_size // 2)
    std_local = torch.sqrt(torch.clamp(mean_square_local - mean_local ** 2, min=eps**2))

    return (x - mean_local) / (std_local + eps), mean_local, std_local
