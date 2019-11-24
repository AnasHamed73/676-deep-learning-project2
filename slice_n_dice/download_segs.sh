#!/bin/bash
# This script downloads the segmentation masks identified in the file "seg_ids"
# for each of the sources identified in the "srcs"dir".
# Like the segmentation ids, the masks themselves are also downloaded via
# a GET request to the REST API of ISIC

# names of the directories in "SRCS_DIR" to process 
SRCS_FILE="/home/kikuchio/Documents/courses/deep-learning/project2/unprocessed"
SRCS_DIR="/home/kikuchio/Documents/courses/deep-learning/project2/originals/working"
SEG_IDS_FILENAME="seg_ids"
BASE_URL="https://isic-archive.com/api/v1/segmentation"
URL_MASK_ENDPOINT_ID="mask"
MASK_POSTFIX="_mask.png"

for src in $(cat $SRCS_FILE); do
	src_dir="${SRCS_DIR}/${src}"
	ids_file="${src_dir}/${SEG_IDS_FILENAME}"
	for line in $(cat $ids_file); do
		name=$(echo $line | cut -d ',' -f1)
		seg_id=$(echo $line | cut -d ',' -f2)
		echo "requesting mask for file ${name} using seg_id ${seg_id}"
    curl -X GET -o "${src_dir}/${name}${MASK_POSTFIX}" --header 'Accept: image/png' "${BASE_URL}/${seg_id}/${URL_MASK_ENDPOINT_ID}"
	done
done
