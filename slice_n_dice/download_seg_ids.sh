#!/bin/bash
# This script downloads the segmentation mask ids of the images identified
# by their ids in the file "img_ids" for each of the source dirs in "SRC_NAMES_FILE".
# The ids are downloaded via a request to the REST API provided by ISIC.
# The segmentation ids are stored in a CSV file along with the name of the image
# (without the file extension) in the following order: ${image_name},${segmentation_id}

SRC_NAMES_FILE="/home/kikuchio/Documents/courses/deep-learning/project2/unprocessed"
BASE_URL="https://isic-archive.com/api/v1/segmentation?limit=50&sort=created&sortdir=-1&imageId="

for i in $(cat $SRC_NAMES_FILE); do
    IMG_IDS="/home/kikuchio/Documents/courses/deep-learning/project2/originals/working/${i}/img_ids"
    OUTPUT="/home/kikuchio/Documents/courses/deep-learning/project2/originals/working/${i}/seg_ids"
    
    i=0
    num_imgs=$(wc -l ${IMG_IDS})
    for img in $(cat ${IMG_IDS}); do
    	echo "${i}/${num_imgs}"
    	img_id=$(echo $img | cut -d ',' -f1)
    	img_name=$(echo $img | cut -d ',' -f2)
    	out=$(curl -X GET  --header 'Accept: application/json' "${BASE_URL}${img_id}" | cut -d : -f2 | tr -d ' ' | cut -d "\"" -f 2)
    	if [ $out == "[]" ]; then
				echo "no seg mask found for $img"
    	  ((i=i+1))
    		continue
    	else
    		echo "${img_name},${out}" >> "$OUTPUT"
    	fi
    	((i=i+1))
    done
done
