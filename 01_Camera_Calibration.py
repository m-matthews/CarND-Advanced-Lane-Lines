# -*- coding: utf-8 -*-
"""
Advanced Lane Line Project - Camera Calibration

Author: Michael Matthews
"""

import os
import glob
import numpy as np
import cv2
import yaml

CHESSBOARD_X = 9
CHESSBOARD_Y = 6

if __name__ == '__main__':
    objpoints = [] # 3D points in real world space.
    imgpoints = [] # 2D points in image plane.

    # Generate the list of camera calibration filenames.
    imagefiles = sorted(glob.glob("camera_cal/calibration*.jpg"))

    # Prepare object point structures.
    objp = np.zeros((CHESSBOARD_Y*CHESSBOARD_X, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_X, 0:CHESSBOARD_Y].T.reshape(-1, 2) # X, Y coordinates.

    # Loop through the list of calibration file names.
    for imagefile in imagefiles:
        img = cv2.imread(imagefile)

        # Convert to grayscale.
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Detect chessboard corners.
        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_X, CHESSBOARD_Y), None)

        # If corners are detected, add to the lists of object and image points.
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            # Preview detection is working correctly.
            img = cv2.drawChessboardCorners(img, (CHESSBOARD_X, CHESSBOARD_Y), corners, ret)
            cv2.imwrite(os.path.join("output_images", "ccal_" + imagefile.split("/")[-1]), img)
        else:
            print("Unable to find chessboard corners for '{}'".format(imagefile))

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save the camera calibration details to a YAML format for later processes.
    with open("01_Camera_Calibration.yaml", "w") as f:
        yaml.dump({'ret': ret, 'mtx': mtx.tolist(), 'dist': dist.tolist()}, stream=f)

    # Convert two images for documentation in writeup.md.
    for testfile in ["camera_cal/calibration1.jpg", "test_images/test1.jpg"]:
        img = cv2.imread(testfile)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite(os.path.join("output_images", "ccal_out_" + testfile.split("/")[-1]),
                    undist)
