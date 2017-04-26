# -*- coding: utf-8 -*-
"""
Advanced Lane Line Project - Pipeline

Author: Michael Matthews
"""

import os
import glob
import numpy as np
import cv2
import yaml
from moviepy.editor import VideoFileClip


class Line:
    """Class to contain details of a given line.

    This class accepts binary image(s) and detects pixels likely to represent
    the lane line ('left'|'right').  The resulting pixels are fitted with a
    polynomial expression.
    Subsequent images are fitted, taking into account the previous images to
    improve lane detection.

    Attributes:
        side: String defining the lane side ('left'|'right').
        nwindows: Number of sliding windows for initial line detection process.
        margin: +/- number of x-pixels to scan in each window.
        minpix: Number of pixels required to recenter window
        niterations: Number of image processing iterations to keep.
        color: Color used to represent this 'side' in displays.
        detected: Was the line detected in the last iteration?
        base: x-pixel location of the base of the line closest to the vehicle.
        base_offset: offset of base from the centre of the image.
        current_fit: polynomial coefficients for the most recent fit.
        iterx: x coordinates detected to be part of the line.  One list for
               each iteration (eg: [[iter5-pts], [iter4-pts], ...])
        itery: y coordinates detected to be part of the line.  One list for
               each iteration (eg: [[iter5-pts], [iter4-pts], ...])
        curverad: Current estimated curve radius.
    """

    nwindows = 9
    margin = 100
    minpix = 50
    niterations = 5

    def __init__(self, side):
        """Initialises the Line class with the 'side' to be processed."""
        self.side = side
        self.color = [255, 0, 0] if side == 'left' else [0, 0, 255]

        self.detected = False
        self.base = None
        self.base_offset = None
        self.current_fit = [np.array([False])]  
        self.iterx = []
        self.itery = []
        self.curverad = None 

        return

    def reset(self):
        """
        Called from Lane if the lines are not valid and the search should
        begin again with the windowing functions.
        """
        self.detected = False
        return

    def process(self, binary, overlay_img, hud_img):
        """Process the binary image to determine the line position.
    
        The binary image is examined to determine the line position and
        polynomial representation.  The result is displayed onto the 
        overlay_img (which will be flattened back onto the road surface) and
        also the hud_img which will be displayed in the top-right as a
        representation of this line processing.
    
        Args:
            binary: The binary image pre-processed by the Lane class.
            overlay_img: Image to be updated with the relevant points used to
                         detect this line.
            hud_img: Image to be displayed in top-right corner to show inner
                     working of this line detection process.
    
        Returns:
            None.
        """
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Does the initial line base need to be found?
        if self.detected == False:
            window_scan = True
            # Take a histogram of the bottom third of the image
            histogram = np.sum(binary[binary.shape[0]//3:,:], axis=0)

            # Find the peak of the histogram which will be the starting point
            # for the lane line.
            midpoint = np.int(histogram.shape[0]/2)
            if self.side == 'left':
                x_current = np.argmax(histogram[:midpoint])
            else:
                x_current = np.argmax(histogram[midpoint:]) + midpoint
        
            # Set height of windows.
            window_height = np.int(binary.shape[0]/self.nwindows)

            # Create empty lists to receive lane pixel indices
            lane_inds = []
            
            # Step through the windows one by one
            for window in range(self.nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary.shape[0] - (window+1)*window_height
                win_y_high = binary.shape[0] - window*window_height
                win_x_low = x_current - self.margin
                win_x_high = x_current + self.margin

                # Draw the windows on the visualization image
                cv2.rectangle(hud_img,(win_x_low//4,win_y_low//4),
                                      (win_x_high//4,win_y_high//4),
                                      (0,255,0), 1) 
                
                # Identify the nonzero pixels in x and y within the window
                good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

                # Append these indices to the lists
                lane_inds.append(good_inds)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_inds) > self.minpix:
                    x_current = np.int(np.mean(nonzerox[good_inds]))
            
            # Concatenate the arrays of indices
            lane_inds = np.concatenate(lane_inds)
            
            # The initial line detection is not required for future calls.
            self.detected = True
        else:
            window_scan = False
            # Assume you now have a new warped binary image 
            # from the next frame of video (also called "binary_warped")
            # It's now much easier to find line pixels!
            lane_inds = ((nonzerox > (self.current_fit[0]*(nonzeroy**2) +
                                      self.current_fit[1]*nonzeroy +
                                      self.current_fit[2] - self.margin)) &
                         (nonzerox < (self.current_fit[0]*(nonzeroy**2) +
                                      self.current_fit[1]*nonzeroy +
                                      self.current_fit[2] + self.margin)))

        # Remove the oldest iteration if the required limit is reached.
        if len(self.iterx)>=self.niterations:
            self.iterx.pop(0)
            self.itery.pop(0)
            
        # Extract line pixel positions and update the latest iteration values.
        self.iterx.append(nonzerox[lane_inds])
        self.itery.append(nonzeroy[lane_inds])

        # Concatenate all of the points for all iterations into a single list.
        x = np.concatenate(self.iterx)
        y = np.concatenate(self.itery)
        # Build a list of weights based on the iterations.  This is 1.0 for
        # the latest, and 0.8, 0.8, 0.4 and 0.2 for previous iterations.
        w = np.concatenate([np.repeat((i+1)/len(self.iterx), len(self.iterx[i]))
                            for i in range(len(self.iterx))])
        
        # Fit a second order polynomial to the points, using the weighted values.
        self.current_fit = np.polyfit(y, x, 2, w=w)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0] )
        current_fitx = self.current_fit[0]*ploty**2 + self.current_fit[1]*ploty + self.current_fit[2]
        # Restrict the points to within the screen.
        current_fitx = np.maximum(0, np.minimum(binary.shape[1]-1, current_fitx))

        # Determine the base of the line, and offset to centre.
        self.base = current_fitx[-1]
        self.base_offset = self.base - binary.shape[1]//2

        if window_scan == False:
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            line_window1 = np.array([np.transpose(np.vstack([current_fitx-self.margin, ploty]))])
            line_window2 = np.array([np.flipud(np.transpose(np.vstack([current_fitx+self.margin, ploty])))])
            line_pts = np.hstack((line_window1, line_window2))
            
            # Draw the lane onto the warped HUD image.
            cv2.fillPoly(hud_img, np.int_([line_pts])//4, (0, 255, 0))

        hud_img[nonzeroy[lane_inds]//4, nonzerox[lane_inds]//4] = self.color
        for i in range(0, len(self.iterx)-1):
            fadecolor = np.maximum(self.color,int(255/self.niterations*(self.niterations-i-1)))
            hud_img[self.itery[i]//4, self.iterx[i]//4] = fadecolor

        overlay_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = self.color

        # Draw the contributing points in the color selected for this line.
        hud_img[nonzeroy[lane_inds]//4, nonzerox[lane_inds]//4] = self.color
        hud_img[nonzeroy[lane_inds]//4, nonzerox[lane_inds]//4] = self.color

        # Draw the polynomial in yellow.
        hud_img[np.array(ploty//4, dtype=np.int32),
                np.array(current_fitx//4, dtype=np.int32)] = [255, 255, 0]
        
        return


class Lane:
    """Class to contain lane details of the left and right line.

    This class accepts image and uses the Line class to track the position of
    the left and right line making up the current lane on the road.

    Attributes:
        colspace: Source image style ('rgb'|'bgr')
        ksize: Kernel size to use in Sobel operations.
        niterations: Number of image processing iterations to keep for
                     determining the average road radius to remove noise.
        firstscan: Detect if this is the first image scanning operation of a
                   movie sequence.
        iterfits: List of curve fit polynomial from each iteration.
        left_line: Instance of the Line class for the left lane line.
        right_line: Instance of the Line class for the right lane line.
        mtx: Camera calibration details.
        dist: Camera calibration details.
        M: Perspective Transformation details.
        xm_per_pix: meters per pixel in x dimension
        ym_per_pix: meters per pixel in y dimension
        debug: Debugging status.
    """
    niterations = 12
        
    # Conversions in x and y from pixels space to meters
    # Based on file: output_images/perspective_transform_out.jpg:
    #   The gap between line centres (3.7m) is ~750 pixels.
    #   The white line (3m) is ~100 pixels high.
    xm_per_pix = 3.7/750
    ym_per_pix = 3/100

    def __init__(self, colspace="rgb", ksize=3, debug=False):
        """Initialise the Line instance.
    
        Args:
            colspace: Color space for input images.
            ksize: Kernal Size to use in Sobel operations.
            debug: Flag to save intermediate processing images to disk.
        """

        # Record the expected color space to be used by this instance.
        # OpenCV images are normally in the non-standard BGR space.
        self.colspace = colspace
        # Sobel kernel size
        self.ksize = ksize
        # Is this the initial scan?
        self.firstscan = True
        # List of centre line fit polynomials.
        self.iterfits = []
        # Keep debugging status.
        self.debug = debug

        # Import Camera Calibration values created in 01_Camera_Calibration.py.
        with open("01_Camera_Calibration.yaml", "r") as f:
            ydata = yaml.load(stream=f)
            self.mtx = np.array(ydata['mtx'])
            self.dist = np.array(ydata['dist'])
        
        # Import Perspective Transform values created in 02_Perspective_Transform.py.
        with open("02_Perspective_Transform.yaml", "r") as f:
            ydata = yaml.load(stream=f)
            self.M = np.array(ydata['M'])

        self.left_line = Line("left")
        self.right_line = Line("right")

        return

    def show(self, image, winname="image", binary=False):
        """Display the image in a window for debugging purposes."""
        if binary:
            image=image*255
        cv2.namedWindow(winname)
        cv2.imshow(winname, image)
        cv2.waitKey()
        return

    def _preprocess(self, image):
        """Preprocess the image to create consistent color space."""
        if self.colspace == "bgr":
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return image

    def _binarythreshold(self, image, thresh):
        """Perform thresholding limits on a binary image."""
        binary_output = np.zeros_like(image)
        binary_output[(image >= thresh[0]) & (image <= thresh[1])] = 1
        return binary_output

    def _abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, thresh=(0, 255)):
        """Calculate the directional threshold gradient.
    
        Args:
            image: Input grayscale image.
            orient: Orientation for scan ('x'|'y').
            sobel_kernel: Kernel size for Sobel function.
            thresh: Threshold in form (min, max).
    
        Returns:
            Binary image based on supplied threshold.
        """
        sobel = cv2.Sobel(image, cv2.CV_64F, 1 if orient=="x" else 0,
                                             0 if orient=="x" else 1,
                                             ksize = sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        return self._binarythreshold(scaled_sobel, thresh)
    
    def _mag_thresh(self, image, sobel_kernel=3, thresh=(0, 255)):
        """Calculate the gradient magnitude.
    
        Args:
            image: Input grayscale image.
            sobel_kernel: Kernel size for Sobel function.
            thresh: Threshold in form (min, max).
    
        Returns:
            Binary image based on supplied threshold.
        """
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        abs_sobel = np.sqrt(np.square(sobelx)+np.square(sobely))
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        return self._binarythreshold(scaled_sobel, thresh)
    
    def _dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):
        """Calculate the gradient direction.
    
        Args:
            image: Input grayscale image.
            sobel_kernel: Kernel size for Sobel function.
            thresh: Threshold in form (min, max).
    
        Returns:
            Binary image based on supplied threshold.
        """
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
        return self._binarythreshold(dir_sobel, thresh)

    def _combined_threshold(self, image, x_thresh, y_thresh, m_thresh, d_thresh):
        """Calculate the combined threshold.
    
        Args:
            image: Input grayscale image.
            x_thresh: Threshold to apply to x directional gradient.
            y_thresh: Threshold to apply to y directional gradient.
            m_thresh: Threshold to apply to magnitude gradient.
            d_thresh: Threshold to apply to gradient direction.
    
        Returns:
            Binary image from combined thresholds.
        """
        gradx = self._abs_sobel_thresh(image, orient='x', sobel_kernel=self.ksize,
                                       thresh=x_thresh)
        grady = self._abs_sobel_thresh(image, orient='y', sobel_kernel=self.ksize,
                                       thresh=y_thresh)
        mag_binary = self._mag_thresh(image, sobel_kernel=self.ksize,
                                      thresh=m_thresh)
        dir_binary = self._dir_threshold(image, sobel_kernel=self.ksize,
                                         thresh=d_thresh)

        combined_binary = np.zeros_like(dir_binary)
        combined_binary[((gradx == 1) & (grady == 1)) |
                        ((mag_binary == 1) & (dir_binary == 1))] = 1

        return combined_binary

    def _thresholds(self, image):
        """Calculate thresholds for all required channels.
    
        Args:
            image: Input grayscale image.
    
        Returns:
            Binary image from combined thresholds.
        """

        # Convert to HLS color space and separate the S channel.
        s_channel = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:,:,2]
        # The Red channel is a suitable alternative to gray.
        r_channel = image[:,:,0]

        s_combined_binary=self._combined_threshold(s_channel, x_thresh=(100, 255),
                                                              y_thresh=(100, 255),
                                                              m_thresh=(30, 100),
                                                              d_thresh=(0.7, 1.3))
        r_combined_binary=self._combined_threshold(r_channel, x_thresh=(20, 100),
                                                              y_thresh=(100, 200),
                                                              m_thresh=(60, 150),
                                                              d_thresh=(0.7, 1.3))

        # Initial scan only uses the s channel.  After this the combination
        # with red is suitable to continue lane line detection.
        if self.firstscan == True:
            combined_binary = s_combined_binary
            self.firstscan = False
        else:
            combined_binary = np.zeros_like(s_combined_binary)
            combined_binary[((s_combined_binary == 1) | (r_combined_binary == 1))] = 1

        # Export thresholding images if the debug flag is set.
        if self.debug:
            cv2.imwrite("./output_images/threshold_s_channel.jpg", s_channel)
            cv2.imwrite("./output_images/threshold_r_channel.jpg", r_channel)
            cv2.imwrite("./output_images/threshold_s_binary.jpg", s_combined_binary*255)
            cv2.imwrite("./output_images/threshold_r_binary.jpg", r_combined_binary*255)
        
        return combined_binary

    def _findlane(self, binary, image):
        """Using the supplied images determine the left and right lane lines.
    
        Args:
            binary: Binary image of potential lane line pixels.
            image: Original image frame (distorted).
    
        Returns:
            Updated image for display with lane markings and additional
            information.
        """

        # HUD image will contain the top-down display with line detection
        # information.  It will be 1/16 scale in the top-right of the final
        # image projection.
        hud_img = cv2.warpPerspective(image, self.M,
                                      (image.shape[1],image.shape[0]),
                                      flags=cv2.INTER_LINEAR)
        hud_img = cv2.resize(hud_img, (0, 0), fx=0.25, fy=0.25)
        hud_overlay_img = np.zeros_like(hud_img)

        # Overlay image is used to project the detected lanes back onto the
        # original image/road surface.
        overlay_img = np.zeros_like(image)

        # Process the new input image with each line.
        self.left_line.process(binary, overlay_img, hud_overlay_img)
        self.right_line.process(binary, overlay_img, hud_overlay_img)

        # Add the final overlay polygon using the output of both line detections.
        overlay_poly = []
        for y in range(0, overlay_img.shape[0]+1, 36):
            overlay_poly.append((self.left_line.current_fit[0]*y**2 +
                                 self.left_line.current_fit[1]*y +
                                 self.left_line.current_fit[2], y))
        for y in range(overlay_img.shape[0], -1, -36):
            overlay_poly.append((self.right_line.current_fit[0]*y**2 +
                                 self.right_line.current_fit[1]*y +
                                 self.right_line.current_fit[2], y))

        cv2.fillPoly(overlay_img, [np.array(overlay_poly, dtype=np.int32)], (0, 255, 0))
        
        # Add the average of the two Lines into the current fit.
        self.iterfits.append(np.average([self.left_line.current_fit,
                                         self.right_line.current_fit], axis=0))
        if len(self.iterfits)>self.niterations:
            self.iterfits.pop(0)

        # Create average fit of the all retained iterations.
        avgfit = np.average(self.iterfits, axis=0)
        ploty = np.linspace(0, overlay_img.shape[0]-1, overlay_img.shape[0] )
        plotx = avgfit[0]*ploty**2 + avgfit[1]*ploty + avgfit[2]
        # Restrict the points to within the screen.
        plotx = np.maximum(1, np.minimum(overlay_img.shape[1]-2, plotx))

        overlay_img[np.array(ploty, dtype=np.int32),
                    np.array(plotx-1, dtype=np.int32)] = [0, 0, 0]
        overlay_img[np.array(ploty, dtype=np.int32),
                    np.array(plotx, dtype=np.int32)] = [0, 0, 0]
        overlay_img[np.array(ploty, dtype=np.int32),
                    np.array(plotx+1, dtype=np.int32)] = [0, 0, 0]

        # Calculate the new radii of curvature in meters.
        curverad = ((1 + (2*avgfit[0]*np.max(ploty)*self.ym_per_pix + avgfit[1])**2)**1.5) / np.absolute(2*avgfit[0])

        # Warp the overlay image back into the original perspective.        
        overlay_img = cv2.warpPerspective(overlay_img, self.M,
                                          (overlay_img.shape[1],overlay_img.shape[0]),
                                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        image = cv2.addWeighted(image, 1.0, overlay_img, 0.4, 0)

        # Add the HUD image into the top right corner of the display.
        hud_img = cv2.addWeighted(hud_img, 1.0, hud_overlay_img, 0.4, 0)
        yoff = 10
        xoff = image.shape[1] - hud_img.shape[1] - 10
        image[yoff:yoff+hud_img.shape[0], xoff:xoff+hud_img.shape[1]] = hud_img
        cv2.rectangle(image, (xoff-2, yoff-2), (xoff+hud_img.shape[1]+1, yoff+hud_img.shape[0]+1),
                      (255, 255, 255), 2)

        # Display curve radius on the output image.
        self._text(image, "Curve Radius: " +
                          ("Straight." if curverad>6000 else "{:.0f}m.".format(curverad)), 0)
        
        # Display vehicle left/right offset on the output image.
        veh_offset = (self.left_line.base_offset*self.xm_per_pix +
                      self.right_line.base_offset*self.xm_per_pix)/2
        if abs(veh_offset)<0.02:
            self._text(image, "Vehicle: Centred.", 1)
        else:
            self._text(image, "Vehicle: {:.2f}m {} of centre.".format(abs(veh_offset), "right" if veh_offset<0 else "left"), 1)
        
        return image

    def _text(self, image, text, line, size=1.0):
        """Display text on an image."""
        cv2.putText(image, text, (10, 35+int(line*35)), cv2.FONT_HERSHEY_SIMPLEX,
                    size, (255, 255, 255), thickness=2 if size==1.0 else 1)
        return
        
    def process(self, image):
        """Process a given image to find the left and right lane lines.
    
        Args:
            image: Original image frame.
    
        Returns:
            Updated image for display with lane markings and additional
            information.
        """

        # Perform pre-processing of the input image.
        image = self._preprocess(image)

        # Perform the thresholding before any distortion to ensure the
        # input to thresholding is as clean as possible.
        bimage = self._thresholds(image)

        # Remove camera distortion.
        image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        bimage = cv2.undistort(bimage, self.mtx, self.dist, None, self.mtx)

        # Flatten the binary image to appear as a top-down view.
        bwarped = cv2.warpPerspective(bimage, self.M,
                                      (bimage.shape[1],bimage.shape[0]),
                                      flags=cv2.INTER_LINEAR)
        
        out_img = self._findlane(bwarped, image)
        
        # When returning an image, return it in the same format as provided.
        if self.colspace == "bgr":
            out_img = cv2.cvtColor(out_img,cv2.COLOR_RGB2BGR)

        # Ensure debugging images are only exported once.
        self.debug = False

        return out_img


if __name__ == '__main__':
    # Process the list of test_images using the Lane class and export to
    # the output_images.
    imagefiles = sorted(glob.glob("test_images/*.jpg"))
    for imagefile in imagefiles:
        print("Processing", imagefile)
        lane = Lane(colspace="bgr", ksize=3)
        img = cv2.imread(imagefile)
        img = lane.process(img)
        cv2.imwrite(os.path.join("output_images", "test_" + imagefile.split("/")[-1]), img)

    # Process a single image with the 'debug' flag set to extract the binary
    # thresholding data for the final writeup.md.
    print("Processing test_images/test4.jpg")
    lane = Lane(colspace="bgr", ksize=3, debug=True)
    img = cv2.imread("test_images/test4.jpg")
    img = lane.process(img)
    
    # Process a set of images to demonstrate weighted line iterations for the
    # final writeup.md.
    lane = Lane(colspace="bgr")
    imagefiles = sorted(glob.glob("output_images/project_video_*.jpg"))
    for imagefile in imagefiles:
        print("Processing", imagefile)
        img = cv2.imread(imagefile)
        img = lane.process(img)
        cv2.imwrite(os.path.join("output_images", "test_" + imagefile.split("/")[-1]), img)
    
    # Create an instance of the Lane class to process the project_video.
    lane = Lane(colspace="rgb")
    def process_image(image):
        return lane.process(image)
    
    if 1:
        videofile = 'project_video'
        clip = VideoFileClip(videofile + '.mp4')
#        # Test the first 5 seconds of the movie.
#        clip = VideoFileClip(videofile + '.mp4').set_end((0, 5))
        project_clip = clip.fl_image(process_image)
        project_clip.write_videofile('output_images/' + videofile + '_output.mp4', audio=False)
#        # Write the individual video frames as images for testing.
#        clip.write_images_sequence('output_images/video/' + videofile + '_output_%d.jpg')
