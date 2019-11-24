#!/bin/bash
# This script gathers all the segmented images, along with their 
# corresponding json metadata files, from different sources
# in a single directory

JSONS_DIR="/home/kikuchio/Documents/courses/deep-learning/project2/originals/ISIC-images"
SEG_DIR="/home/kikuchio/Documents/courses/deep-learning/project2/originals/working"
SRCS_FILE="/home/kikuchio/Documents/courses/deep-learning/project2/unprocessed"
OUTPUT_DIR="/home/kikuchio/Documents/courses/deep-learning/project2/all"
SEG_DIR_NAME="seg"

for src in $(cat $SRCS_FILE); do
	echo "processing $src"
	seg_src="${SEG_DIR}/${src}/${SEG_DIR_NAME}"
	jsons_src="${JSONS_DIR}/${src}"
	for img in $(ls $seg_src); do
		echo "processing ${img}"
		img_name=$(echo "$img" | cut -d . -f1)
		mv -n "${seg_src}/${img_name}.jpg" "${OUTPUT_DIR}"
		mv -n "${jsons_src}/${img_name}.json" "${OUTPUT_DIR}"
	done
done

	


