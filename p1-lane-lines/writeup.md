# **Finding Lane Lines on the Road**

[//]: # (Image References)

[original]: ./pipeline_images/original.png "Original"
[gray]: ./pipeline_images/gray.png "Grayscale"
[blur_gray]: ./pipeline_images/blur_gray.png "Blur Grayscale"
[edges]: ./pipeline_images/edges.png "Edges"
[masked]: ./pipeline_images/masked.png "Masked"
[lines]: ./pipeline_images/lines.png "Lines"
[draw]: ./pipeline_images/draw.png "Draw"

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. The followings showed step-by-step results from original
image to final result.

![Original][original]

Step 1. Convert original image from RGB to grayscale.

![Grayscale][gray]

Step 2. Apply Gaussian blur to the grayscale image.

The kernel size is 5x5.

![Blur Grayscale][blur_gray]

Step 3. Use Canny edge detection to find the edges.

The low threshold is 50, and high threshold is 150.

![Edges][edges]

Step 4. Mask the region of interest by a trapezoid.

The trapezoid was formed from 60% of the height to the bottom of the image, and the
upper two vertices were picked from 20 pixels from the center of the top edge
and the bottom two vertices were the leftmost and rightmost points.

![Masked][masked]

Step 5. Apply Hough transformation to get line segments.

Step 6. Compute the slopes and intercepts of line segments and take the median
of them to form two lane lines equations.

Step 7. Solve the equations to get the endpoints of lane lines and draw them
on the original image.

![Lines][lines]

In order to draw a single line on the left and right lanes, I modified the
draw_lines() function by taking the medians of the slopes and intercepts of the
segments from Hough transform. Based on the slopes and intercepts, we could
form the equations of the left and right lanes. By solving the equations of
these two lines and the horizontal line of 60% of the image height and the
bottom line, four intersection points could be derived to draw the lane lines.

![Draw][draw]

### 2. Identify potential shortcomings with your current pipeline

* The lane lines out of the region of interest would not be detected. Since this
region was created by observing the videos beforehand, it only worked for the
lane lines in this region. Other objects out of the region would be filtered out.
* The current pipeline could not draw curved lines. When the vehicle started to
make an U-turn, the detected lane lines would be incorrect.
* The lines drew by the pipeline could deviate from the real line if the vehicle
drives on dirty roads. The Hough transform may detect many noisy lines, and the
median of them would drift from the reasonable range.

### 3. Suggest possible improvements to your pipeline

* Search the region of interest automatically instead of calculating it by human
to solve the lane lines out of region issue.
* Apply curve detection to make it also work on turning.
* Use two or more regions of interest instead of one to decrease the noisy lines
and improve the accuracy of line drawing.
