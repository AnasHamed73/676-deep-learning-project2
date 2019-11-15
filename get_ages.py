#!/usr/bin/env python
# This script partitions the lesions into benign and malignant directories based
# on the meta data contained in the json files that accompany each sample

import os
import json
import glob
import shutil

base_dir = "/home/kikuchio/Documents/courses/deep-learning/project2/all"
src_path = "/home/kikuchio/Documents/courses/deep-learning/project2/"
malignant_dir = "/home/kikuchio/Documents/courses/deep-learning/project2/samples/1"
benign_dir = "/home/kikuchio/Documents/courses/deep-learning/project2/samples/0"

######MAIN

os.chdir(base_dir)
os.makedirs(malignant_dir, exist_ok=True)
os.makedirs(benign_dir, exist_ok=True)

ben_cnt = mal_cnt = unk_cnt = 0
for src in [malignant_dir, benign_dir]:
    os.chdir(src)
    print("current dir: ", src)
    jsons = glob.glob("*.json")
    for i, jf in enumerate(jsons, 0):
        with open(jf, 'r') as doc:
            data = json.load(doc)
            name = data['name']
            age = data['meta']['clinical']['age_approx']
            print(age)

