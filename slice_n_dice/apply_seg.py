#!/usr/bin/env python
# This script applies the segmentation masks (if they exist) for each image

import os
import glob
import cv2
import numpy as np

SRCS_DIR = "/home/kikuchio/Documents/courses/deep-learning/project2/originals/ISIC-images"
SRC_DIR_NAMES_FILE = "/home/kikuchio/Documents/courses/deep-learning/project2/unprocessed"
MASK_DIR = "/home/kikuchio/Documents/courses/deep-learning/project2/originals/working/"
SEG_DIR = "/home/kikuchio/Documents/courses/deep-learning/project2/originals/working/"

SEG_DIR_NAME = "seg"
MASK_POSTFIX = "_mask.png"

def apply_mask(img_path, mask_path, output_file):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)
    print("img: ", img)
    print("mask: ", mask)
    i, j = np.where(mask)
    indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                          np.arange(min(j), max(j) + 1),
                          indexing='ij')
    sub_image = img[indices]
    cv2.imwrite(output_file, sub_image)


######MAIN

with open(SRC_DIR_NAMES_FILE, 'r') as src_names:
    for src in src_names.read().splitlines():
        src_path = os.path.join(SRCS_DIR, src)
        output_dir = os.path.join(SEG_DIR, src, SEG_DIR_NAME)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(src_path)
        imgs = glob.glob("*.jpg")
        num_imgs = len(imgs)
        print("current dir: ", src)
        for i, img in enumerate(imgs, 0):
            print(f"iter {i}/{num_imgs}")
            img_name = img.split(".")[0]
            img_path = os.path.join(src_path, img_name + ".jpg")
            mask_path = os.path.join(MASK_DIR, src, img_name + MASK_POSTFIX)
            output_path = os.path.join(output_dir, img_name + ".jpg")
            if not os.path.exists(mask_path):
                print(f"skipping {img_name}: no mask found")
                continue
            if os.path.exists(output_path):
                print(f"{img_name} has already been processed")
                continue
            try:
                apply_mask(img_path, mask_path, output_path)
            except ValueError as e:
                print(f"something went wrong with {img_name}: {e}")

