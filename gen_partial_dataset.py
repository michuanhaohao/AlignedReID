import cv2
import os
import numpy as np

data_root = "/home/csc302/workspace/luohao/code/AlignedReID/data/market1501/query"
gen_root = "/home/csc302/workspace/luohao/code/AlignedReID/data/market1501_partial/query"

def random_crop(img, sample_rate=0.6):
    h,w = img.shape[:2]
    sh = np.random.randint(sample_rate*h, h*0.9,1)[0]
    bh = np.random.randint(0, h-sh, 1)[0]
    img = img[bh:sh+bh,:,:]
    img = cv2.resize(img, (w,h))
    return img


for image_name in os.listdir(data_root):
    if image_name[-3:] != 'jpg':
        continue
    img_path = os.path.join(data_root, image_name)
    img = cv2.imread(img_path)
    img = random_crop(img)
    save_path = os.path.join(gen_root, image_name)
    cv2.imwrite(save_path, img)