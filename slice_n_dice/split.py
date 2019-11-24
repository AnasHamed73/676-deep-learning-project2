#!/usr/bin/env python
# This script divides the already partitioned data into a train/validation/test
# split. The split is determined in the the "data_splits" file

import os
import json
import glob
import shutil
import random

base_dir = "/home/kikuchio/courses/dl/project2/samples"
#base_dir = "/home/kikuchio/Documents/courses/deep-learning/project2/samples"
data_splits_path = "/home/kikuchio/courses/dl/project2/splits"
malignant_dir = "/home/kikuchio/courses/dl/project2/all/1"
benign_dir = "/home/kikuchio/courses/dl/project2/all"


def get_samples_and_jsons(path):
    os.chdir(path)
    samples = glob.glob("*jpg")
    jsons = glob.glob("*json")
    return samples, jsons


def move_img_and_json(img_full_name, src_dir, target_dir):
    name = img_full_name.split(".")[0]
    img_src_path = os.path.join(src_dir, name + ".jpg")
    json_src_path = os.path.join(src_dir, name + ".json")
    img_target_path = os.path.join(target_dir, name + ".jpg")
    json_target_path = os.path.join(target_dir, name + ".json")
    shutil.move(img_src_path, img_target_path)
    shutil.move(json_src_path, json_target_path)


######MAIN

malignant_samples, malignant_jsons = get_samples_and_jsons(malignant_dir)
benign_samples, benign_jsons = get_samples_and_jsons(benign_dir)

random.shuffle(malignant_samples)
random.shuffle(benign_samples)

with open(data_splits_path, 'r') as data_splits:
    phase = None
    benign_processed = malignant_processed = 0
    for data_split in data_splits.read().splitlines():
        if data_split in ["train","validation", "test"]:
            phase = data_split
        else:
            fields = data_split.split(" ")
            quantity = fields[0]
            diagnosis = fields[1]
            diagnosis_label = "0" if diagnosis == "benign" else "1"
            for i in range(int(quantity)):
                partition = os.path.join(phase, diagnosis_label)
                target_dir = os.path.join(base_dir, partition)

                img_name = benign_samples[benign_processed] \
                        if diagnosis == "benign"\
                        else malignant_samples[malignant_processed]
                img_src_dir = os.path.join(base_dir, diagnosis_label)
                move_img_and_json(img_name, img_src_dir, target_dir)
                if diagnosis == "benign":
                    benign_processed += 1
                else:
                    malignant_processed += 1

