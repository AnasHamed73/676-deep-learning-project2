#!/usr/bin/env python
# This script gathers the ids of the images from their corresponding json files

import os
import json
import glob
import shutil

base_dir = "/home/kikuchio/Documents/courses/deep-learning/project2/ISIC-images"
seg_dir = "/home/kikuchio/Documents/courses/deep-learning/project2/working"
srcs_file="/home/kikuchio/Documents/courses/deep-learning/project2/unprocessed"
img_ids_filename = "img_ids"

######MAIN

os.chdir(base_dir)

with open(srcs_file, 'r') as srcs_file:
    for src in srcs_file.read().splitlines():
        src_dir = os.path.join(base_dir, src)
        os.chdir(src_dir)
        print("current dir: ", src)
        os.makedirs(os.path.join(seg_dir, src), exist_ok=True)
        jsons = glob.glob("*.json")
        for i, jf in enumerate(jsons, 0):
            with open(jf, 'r') as doc:
                data = json.load(doc)
                name = data['name']
                id = data['_id']
                print("processing ", id)
                with open(os.path.join(seg_dir, src, img_ids_filename), 'a+') as img_file:
                    img_file.writelines(id + "," + name + "\n")
    
