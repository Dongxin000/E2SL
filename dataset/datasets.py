from dataset.dataset_rendered import DatasetSL
from pathlib import Path
import re
from glob import glob
from itertools import repeat
import random
# from datasets.data_io import get_transform, read_all_lines, pfm_imread

import sys 
sys.path.append("..")

def GetDataset(path, tgt_res, my_logging, vertical_jitter=1, version="unity_4", debug=False, left_only=False, ):

    root_path = '/data/E2SL_dataset/'
    train_folder_path = root_path + '/train/' # train dataset path
    valid_floder_path = root_path + '/val/' # val dataset path

    len_train = 10000
    train_path_lists = load_data_path(train_folder_path, len_train, True, my_logging, True)
    my_logging.info(f"train length:"+ str(len(train_path_lists[5])))

    len_val = 1000    
    valid_path_lists = load_data_path(valid_floder_path, len_val, True, my_logging, False)
    my_logging.info(f"val length:"+ str(len(valid_path_lists[0])))
    
    datasets = {
        'train': DatasetSL(train_path_lists, True),
        'val': DatasetSL(valid_path_lists, False)
    }
    
    src_res = (1280, 704)
    return datasets, src_res
   
def load_data_path(filepath, len_data, binaryOption, my_logging, train_opt):
    if binaryOption:
        if train_opt:
            left_images = sorted(glob(os.path.join(filepath, os.path.join('left/*.png'))))
        else:
            left_images = sorted(glob(os.path.join(filepath, os.path.join('left_binary/*.png'))))
        
        e2vid = sorted(glob(os.path.join(filepath, os.path.join('e2vid/*.png'))))
        rawprj_images = sorted(glob(os.path.join(filepath, os.path.join('rawprj/*.exr'))))
        rawbg_images = sorted(glob(os.path.join(filepath, os.path.join('rawbg/*.exr'))))
        # right_img = sorted( glob(os.path.join(filepath, os.path.join('right_binary/*.png'))))
        # right_img = sorted(glob(os.path.join(filepath, os.path.join('right_full_binary/*.png'))))
        my_logging.info("---------------binary--------------")
    else:
        left_images = sorted(glob(os.path.join(filepath, os.path.join('left_binary/*.png'))))
        right_img = sorted(glob(os.path.join(filepath, os.path.join('right/*.png'))))            
        my_logging.info("---------------gray--------------")
    
    # right_images = list(repeat(right_img[0], len(left_images)))
    disp_images = sorted(glob(os.path.join(filepath, 'disparity', '*.pfm')))  
    msk_images = sorted(glob(os.path.join(filepath, 'msk', '*.png')))
    
    left_images_path = left_images[:len_data]
    # right_images_path = right_images[:len_data]
    right_images_path = []
    disp_images_path = disp_images[:len_data]
    msk_images_path = msk_images[:len_data]
    rawprj_images_path = rawprj_images[:len_data]
    rawbg_images_path = rawbg_images[:len_data]
    e2vid_images_path = e2vid[:len_data]
    
    path_lists = [left_images_path, right_images_path, disp_images_path, msk_images_path, rawprj_images_path, rawbg_images_path, e2vid_images_path]
    return path_lists