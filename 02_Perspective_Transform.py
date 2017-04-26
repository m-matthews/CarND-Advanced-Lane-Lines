# -*- coding: utf-8 -*-
"""
Advanced Lane Line Project - Perspective Transformation

Author: Michael Matthews
"""

import numpy as np
import cv2
import yaml

if __name__ == '__main__':
    with open("01_Camera_Calibration.yaml", "r") as f:
        camcal = yaml.load(stream=f)
        mtx = np.array(camcal['mtx'])
        dist = np.array(camcal['dist'])

    img = cv2.imread("test_images/straight_lines1.jpg")
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Initial extract was too long and when scaled would quickly leave the side
    # of the screen on corners.
#    src = np.float32([[238,686],[610,439],[669,439],[1068,686]])
#    dst = np.float32([[200,710],[200,0],[1080,0],[1080,710]])

    # Second configuration is shorter in distance, and also is narrower on the
    # projected page so that a greater curve distance can be plotted from the
    # 'top-down' perspective.
#    src = np.float32([[238,686],[573,465],[712,465],[1068,686]])
#    dst = np.float32([[250,710],[250,0],[1030,0],[1030,710]])

    # Final selecting is the original longer distance from first iteration,
    # however using the narrower projection from the second.
#    src = np.float32([[238,686],[610,439],[669,439],[1068,686]])
#    dst = np.float32([[250,710],[250,0],[1030,0],[1030,710]])

    # New configuration is slightly longer in distance so that more line segments
    # can be seen in a single image.
    src = np.float32([[238,686],[581,458],[701,458],[1068,686]])
    dst = np.float32([[250,710],[250,0],[1030,0],[1030,710]])

    M = cv2.getPerspectiveTransform(src, dst)

    # Save the matrix to a YAML format for later processes.
    with open("02_Perspective_Transform.yaml", "w") as f:
        yaml.dump({'M': M.tolist()}, stream=f)

    warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]),
                                             flags=cv2.INTER_LINEAR)

    cv2.polylines(warped, [np.array(dst, dtype=np.int32)], 1, (0, 0, 255))
    cv2.polylines(undist, [np.array(src, dtype=np.int32)], 1, (0, 0, 255))

    # Export images for documentation in writeup.md.
    cv2.imwrite("output_images/perspective_transform_in.jpg", undist)
    cv2.imwrite("output_images/perspective_transform_out.jpg", warped)
