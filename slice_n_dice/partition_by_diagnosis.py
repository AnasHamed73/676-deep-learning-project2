#!/usr/bin/env python
# This script partitions the lesions into benign and malignant directories based
# on the meta data contained in the json files that accompany each sample

import os
import json
import glob
import shutil

BASE_DIR = "/home/kikuchio/Documents/courses/deep-learning/project2/all"
SRC_DIR_NAMES_FILE = "/home/kikuchio/Documents/courses/deep-learning/project2/acceptable"
MALIGNANT_DIR = "/home/kikuchio/Documents/courses/deep-learning/project2/samples/1"
BENIGN_DIR = "/home/kikuchio/Documents/courses/deep-learning/project2/samples/0"

######MAIN

os.chdir(BASE_DIR)
os.makedirs(MALIGNANT_DIR, exist_ok=True)
os.makedirs(BENIGN_DIR, exist_ok=True)

ben_cnt = mal_cnt = unk_cnt = 0
with open(SRC_DIR_NAMES_FILE, 'r') as srcs:
    for src in srcs.read().splitlines():
        current = os.path.join(BASE_DIR, src)
        os.chdir(current)
        print("current dir: ", current)
        jsons = glob.glob("*.json")
        for i, jf in enumerate(jsons, 0):
            print(f"{i}/{len(jsons)}")
            with open(jf, 'r') as doc:
                data = json.load(doc)
                name = data['name']
                if data['meta']['clinical'].get('benign_malignant') is None:
                    unk_cnt += 1
                    continue
                if data['meta']['clinical']['benign_malignant'] == "benign":
                    img_src_path = os.path.join(BASE_DIR, name + ".jpg")
                    json_src_path = os.path.join(BASE_DIR, name + ".json")
                    img_dst_path = os.path.join(BENIGN_DIR, name + ".jpg")
                    json_dst_path = os.path.join(BENIGN_DIR, name + ".json")
                    shutil.move(img_src_path, img_dst_path)
                    shutil.move(json_src_path, json_dst_path)
                    ben_cnt += 1
                elif data['meta']['clinical']['benign_malignant'] == "malignant":
                    img_src_path = os.path.join(BASE_DIR, name + ".jpg")
                    json_src_path = os.path.join(BASE_DIR, name + ".json")
                    img_dst_path = os.path.join(MALIGNANT_DIR, name + ".jpg")
                    json_dst_path = os.path.join(MALIGNANT_DIR, name + ".json")
                    shutil.move(img_src_path, img_dst_path)
                    shutil.move(json_src_path, json_dst_path)
                    mal_cnt += 1
        print(f"{src}: benign cases: {ben_cnt}, malignant cases: {mal_cnt}, unknown cases: {unk_cnt}")
