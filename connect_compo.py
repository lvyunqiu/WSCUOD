import os
import torch
import cv2
from scipy import ndimage
import numpy as np

pred_root = '/home1/lvyunqiu/codes/WSSS/DDT/masks/ddt_mask_ecssd_sig_s16_best/'
pred_list = [f for f in os.listdir(pred_root) if f.endswith('.png')]
save_root = os.path.join(pred_root, 'save_binary')
os.makedirs(save_root, exist_ok=True)

for name in pred_list:
    cur_mask = cv2.imread(os.path.join(pred_root, name), cv2.IMREAD_GRAYSCALE)
    cur_mask[cur_mask<0.4*255] = 0

    objects, num_objects = ndimage.label(cur_mask)
    
    mask = np.zeros(cur_mask.shape)
    for kk in range(1, num_objects+1):
        mask_id = np.where(objects==kk)

        # mask_ori = np.zeros(cur_mask.shape)
        # mask_ori[mask_id[0], mask_id[1]] = cur_mask[mask_id[0], mask_id[1]]
        if len(mask_id[0])< 20:
            continue
        mask[mask_id[0], mask_id[1]] = 255
    cv2.imwrite(os.path.join(save_root, name), mask)