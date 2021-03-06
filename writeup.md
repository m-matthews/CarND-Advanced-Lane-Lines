# Advanced Lane Finding Project

[//]: # (Image References)

[imagecal1]: ./output_images/ccal_calibration3_small.jpg "Chessboard identification 1"
[imagecal2]: ./output_images/ccal_calibration9_small.jpg "Chessboard identification 2"
[imagecal3]: ./output_images/ccal_in_calibration1_small.jpg "Chessboard Calibration Input"
[imagecal4]: ./output_images/ccal_out_calibration1_small.jpg "Chessboard Calibration Output"
[imagecal5]: ./output_images/ccal_in_test1_small.jpg "Road Calibration Output"
[imagecal6]: ./output_images/ccal_out_test1_small.jpg "Road Calibration Output"
[imageper1]: ./output_images/perspective_transform_in_small.jpg "Perspective Transform Input"
[imageper2]: ./output_images/perspective_transform_out_small.jpg "Perspective Transform Output"
[imagevid1]: ./output_images/test_project_video_90_small.jpg "Initial Video Detection"
[imagevid2]: ./output_images/test_project_video_99_small.jpg "Iterative Video Detection"
[imagethr1]: ./output_images/threshold_s_channel_small.jpg "Saturation Channel"
[imagethr2]: ./output_images/threshold_r_channel_small.jpg "Red Channel"
[imagethr3]: ./output_images/threshold_s_binary_small.jpg "Saturation Channel Binary"
[imagethr4]: ./output_images/threshold_r_binary_small.jpg "Red Channel Binary"
[imagetest1]: ./output_images/test_straight_lines1.jpg "Straight Line Detection (1)"
[imagetest2]: ./output_images/test_straight_lines2.jpg "Straight Line Detection (2)"
[imagetest3]: ./output_images/test_test1.jpg "Curved Line Detection (1)"
[imagetest4]: ./output_images/test_test2.jpg "Curved Line Detection (2)"
[imagetest5]: ./output_images/test_test3.jpg "Curved Line Detection (3)"
[imagetest6]: ./output_images/test_test4.jpg "Curved Line Detection (4)"
[imagetest7]: ./output_images/test_test5.jpg "Curved Line Detection (5)"
[imagetest8]: ./output_images/test_test6.jpg "Curved Line Detection (6)"
[videoout1]: ./output_images/project_video_output.mp4 "Processed Video"

## Goals

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Camera Calibration

The Python program to perform the Camera Calibration is [01_Camera_Calibration.py](./01_Camera_Calibration.py).

This program defines the Chessboard X and Y size constants to use in the calibration process, to match the chessboards used in the `./camera_cal` directory.

```python
CHESSBOARD_X = 9
CHESSBOARD_Y = 6
```

Two lists are created to store the chessboard points.  `imgpoints` is appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection using `cv2.findChessboardCorners()`.  `objpoints` will contains the 3D coordinates of the chessboard corners in the real world and is created as the same length as `imgpoints`.

The program is unable to detect the chessboard corners in a number of images as shown in the console output below.  A manual review of these images show that the required corners are not present, and therefore these are ignored for the calibration process.

    Unable to find chessboard corners for 'camera_cal/calibration1.jpg'
    Unable to find chessboard corners for 'camera_cal/calibration4.jpg'
    Unable to find chessboard corners for 'camera_cal/calibration5.jpg'

The detected chessboard corners are visible in the images saved to `/output_images/ccal*.jpg`.  Examples of chessboard detection follow:

| Example 1 | Example 2 |
| --------- | --------- |
| ![alt text][imagecal1] | ![alt text][imagecal2] |

The output `objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients by using the `cv2.calibrateCamera()` function.

The distortion correction was applied to the test image using the `cv2.undistort()` function:

| Chessboard Input | Chessboard Output |
| ---------------- | ----------------- |
| ![alt text][imagecal3] | ![alt text][imagecal4] |

The final parameters are written to the file [01_Camera_Calibration.yaml](./01_Camera_Calibration.yaml) so that subsequent programs can use the same distortion corrections.


## Perspective Transformation

The Python program to derive the Perspective Transformation is [02_Perspective_Transform.py](./02_Perspective_Transform.py).

The selected input image was one of the test images with straight road lines `./test_images/straight_lines1.jpg`.

The image was then undistored using the camera calibration parameters read from the [01_Camera_Calibration.yaml](./01_Camera_Calibration.yaml) file created in the previous step.

A section of line was the selected from the `./output_images/perspective_transform_in.jpg` left and right lanes using GIMP to identify the outer edges of each lane line.  The lowest point (closest to the vehicle) was selected to not be the bottom of the input image to remove parts of the front of the vehicle from the output image.  This can be seen in the 'Perspective Transformation Input' image below.

A number of iterations of the sizing were performed, which can be seen in the commented sizing parameters for `src` and `dst`.  The initial `dst` was too wide, resulting in curved lanes quickly leaving the top of the output perspective transformation image.

The following images show the perspective transformation is performing correctly on a straight road section.

| Perspective Transformation Input | Perspective Transformation Output |
| -------------------------------- | --------------------------------- |
| ![alt text][imageper1] | ![alt text][imageper2] |

The final transformation matrix is written to the file [02_Perspective_Transform.yaml](./02_Perspective_Transform.yaml) so that subsequent programs can import the parameters without regenerating the transformation.


## Pipeline


### Introduction

The Python program to perform the image processing pipeline is [03_Pipeline.py](./03_Pipeline.py).

The pipeline for single images and videos is identical, however the video takes advantage of previous image processing iterations to generate smoothed results.

The program contains two classes:

**Line**: Contains the logic and current state of a lane Line.

**Lane**: Contains the logic and current state of a lane.  A separate instance of **Line** is created for the `left` and `right` lane lines.

The output images (single frame or video) have the following appearance and features:

1) The original image (after Camera Calibration) is used as the image background.  Note that the original undistorted image is not output.
2) Details of the road curvature and vehicle offset are displayed in the top-left.
3) The lane detection process is displayed in the top-right showing the detected pixels for each lane along with a yellow line showing the current polynomial fit to this data.
4) The left and right lines are overlaid back onto the original image with the lane region marked in green.  A centre line is included to show the current estimated polynomial fitting the centre of the lane.

| Initial Line Detection | Iterative Line Detection |
| ---------------- | ----------------- |
| ![alt text][imagevid1] | ![alt text][imagevid2] |


### Process

The **Lane** class is instantiated with the correct color space to use, and then the `process()` method is called for each image.

The first step is to call the `_preprocess()` method which is used to convert to a consistent color space to be used within the system.  OpenCV `imread()` returns images in `BGR` color space, where other packages generally use `RGB`.


#### Thresholding

The next step is to call the `_thresholds()` method which returns a binary image based on the input.  This image attempts to identify the lane markings on the image.  Note that this is performed prior to any attempt to remove camera distortion.

Both the Saturation (`s`) from the HLS color space, and Red (`r`) from the `RGB` color space channels are used in this process.

The `s` channel provides a better detection of the white and yellow lines generally, however I found that the use of the `r` channel assisted in some areas where the image included light and dark areas and the normalised `s` dropped the visibility of some of the line markings.

The `_thresholds()` method creates a combined binary image for both of the `s` and `r` channels as shown in the following example images:

| Saturation (s) Channel | Red (r) Channel |
| ---------------- | ----------------- |
| ![alt text][imagethr1] | ![alt text][imagethr2] |
| ![alt text][imagethr3] | ![alt text][imagethr4] |

The two binary images are then combined into a new single binary image which is the output of the thresholding process.


#### Remove Distortion

The calibration distortion was then applied to both the image and the binary image to ensure the pixel locations are synchronised.  The camera calibration parameters are used from the [01_Camera_Calibration.yaml](./01_Camera_Calibration.yaml) file created in a previous step.

| Road Image Input | Calibrated Road Image Output |
| ---------------- | ---------------------------- |
| ![alt text][imagecal5] | ![alt text][imagecal6] |

The most noticable difference between the images is nearest the edges (for example the white vehicle) however this also straightens the road to enable easier detection and more accurate curve radius information.


#### Lane Line Detection

The `_findlane()` method of the **Lane** class transforms the binary image using the transformation matrix `M` from the [02_Perspective_Transform.yaml](./02_Perspective_Transform.yaml) file created in a previous step.  This image is passed to the `process()` method of the left and right **Line** class instances.

If the line detection has not been performed previously (either a single image or the first frame of a video) then the sliding window method of detection will be applied.  Otherwise the polynomial fit from the previous frame will be used as the windowing technique to find the relevant pixels.  The two examples below show the separate methods in their top-right display.

| Sliding Window Detection | Previous Polynomial Detection |
| ------------------------ | ----------------------------- |
| ![alt text][imagevid1] | ![alt text][imagevid2] |

Note that the second image shows both detected pixels from this iteration along with pixels found in previous iterations.  This can be seen clearly on the right lane where there are dark blue pixels representing the current iteration, and then progressively lighter pixels representing previous frames.  This assists in preventing the line from 'wobbling' as previous data can be used to estimate where the current line should be positioned.  To prevent over-fitting on past results, each iteration is weighted (visible with the lighter colors used in the display).  The current iteration has a weight of 1.0, with previous iterations at 0.8, 0.6, 0.4 and 0.2.

The **Line** class then generates its estimate of the line equation by using the X and Y coordinates of the detected lane line pixels, and the weighting described above:

```python
self.current_fit = np.polyfit(y, x, 2, w=w)
```

The lane detection process is displayed on the `hud_img` image (used in top-right display) and also the `overlay_img` used to project back onto the roadway.


#### Lane Detection

The **Lane** method `_findlane()` uses average polynomial parameters from each **Line** instance which are added to `self.iterfits` to create an ongoing set of recent polynomials for the centre line.  The average of this list provides an estimated centre of the lane which is displayed on the road projection.

The road curvature is derived using the formula suggested in **Udacity Self Driving Car Engineer Nanodegree** lesson **35. Measuring Curvature**.  By using the average of the recent iterations (frames), this ensures a stable representation.

```python
curverad = ((1 + (2*avgfit[0]*np.max(ploty)*self.ym_per_pix + avgfit[1])**2)**1.5) / np.absolute(2*avgfit[0])
```

This section also calculates the position of the base (position closest to the vehicle) of each **Line** instance to determine the vehicle's offset from the centre of the **Lane**.

Both of the previous sections rely on measurements taken from the image after Perspective Transformation.

| Perspective Transformation Output |
| --------------------------------- |
| ![alt text][imageper2]            |

* The gap between line centres should be approximately 3.7m which is used to determine the value for `xm_per_pix`.  Measured on the image this is ~750 pixels.
* The white lane line is approximately 3m in length and is used to determine the value for `ym_per_pix`.  Measured on the image this is ~100 pixels.

```python
xm_per_pix = 3.7/750
ym_per_pix = 3/100
```

#### Output

The final output includes projecting all of the information obtained onto a single image.

| Iterative Line Detection |
| ------------------------ |
| ![alt text][imagevid2]   |


The [project video](./output_images/project_video_output.mp4) successfully identifies the lines and lanes.


## Discussion

The current pipeline correctly identifies the Lane with the `project_video.mp4`.

The pipeline could be enhanced by including:

1) Confirmed the detected line / polynomial is valid.  Currently the components are included to reset the line finding process, however these are not used.  Checking of the following could be included:
  * Curvature is acceptable.
  * The left and right **Line** instances are parallel.
  * The left and right **Line** instances have suitable separation.
  * Position of the calculated **Lane** centre.
2) Additional finetuning of the thresholding processes could also be performed to ensure more challenging environmental conditions are handled.


## Appendix 1: Contents of 'output_images'

The `./output_images` folder contains the following:

* `ccal_*`: Camera Calibration images.
* `perspective_transform_*`: Perspective Transformation images.
* `test_*`: Test images processed from `./test_images` (detailed in **Appendix 2: Test Images**).
  * `test_project_video_*`: Images extracted from the `./project_video.mp4` for use in this `writeup.md`.
* `threshold_*`: Debugging images showing the different channels and binary threshold images.


## Appendix 2: Test Images

The following list shows the identification of the lane lines based on the images supplied in the `./test_images` folder:

| Image                   |
|:-----------------------:|
| ![alt text][imagetest1] |
| ![alt text][imagetest2] |
| * Note that the above image shows a failure to catch the line on the top left, however during the video sequencing the next frame or two would catch this and correct the line. |
| ![alt text][imagetest3] |
| ![alt text][imagetest4] |
| ![alt text][imagetest5] |
| ![alt text][imagetest6] |
| ![alt text][imagetest7] |
| ![alt text][imagetest8] |


