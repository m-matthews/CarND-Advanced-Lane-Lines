#!/bin/bash

# Convert images from the ./output_images folder into smaller versions which can be displayed in the writeup.md.
echo "Image conversion started."

# Cleanup existing small sized images.
rm -f ./output_images/*_small.jpg

# Create 50% sized images.
convert ./output_images/ccal_calibration3.jpg -resize 50% ./output_images/ccal_calibration3_small.jpg
convert ./output_images/ccal_calibration9.jpg -resize 50% ./output_images/ccal_calibration9_small.jpg

convert ./camera_cal/calibration1.jpg -resize 50% ./output_images/ccal_in_calibration1_small.jpg
convert ./output_images/ccal_out_calibration1.jpg -resize 50% ./output_images/ccal_out_calibration1_small.jpg

convert ./test_images/test1.jpg -resize 50% ./output_images/ccal_in_test1_small.jpg
convert ./output_images/ccal_out_test1.jpg -resize 50% ./output_images/ccal_out_test1_small.jpg

convert ./output_images/perspective_transform_in.jpg -resize 50% ./output_images/perspective_transform_in_small.jpg
convert ./output_images/perspective_transform_out.jpg -resize 50% ./output_images/perspective_transform_out_small.jpg

convert ./output_images/test_project_video_90.jpg -resize 50% ./output_images/test_project_video_90_small.jpg
convert ./output_images/test_project_video_99.jpg -resize 50% ./output_images/test_project_video_99_small.jpg

for f in ./output_images/threshold_*.jpg; do
  convert $f -resize 50% ${f%.jpg}_small.jpg;
done

echo "Image conversion complete."

