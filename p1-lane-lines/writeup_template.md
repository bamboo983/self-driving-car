# **Finding Lane Lines on the Road**

[//]: # (Image References)

[original]: ./pipeline_images/original.png "Original"
[gray]: ./pipeline_images/gray.png "Grayscale"
[blur_gray]: ./pipeline_images/blur_gray.png "Blur Grayscale"
[edges]: ./pipeline_images/edges.png "Edges"
[masked]: ./pipeline_images/masked.png "Masked"
[lines]: ./pipeline_images/lines.png "Lines"

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. The followings showed step-by-step results from original
image to final result.

![Original][original]

Step 1. Convert original image from RGB to grayscale

![Grayscale][gray]

Step 2. Apply Gaussian blur to the grayscale image

![Blur Grayscale][blur_gray]

Step 3. Use Canny edge detection to find the edges

![Edges][edges]

Step 4. Mask the region of interest by a trapezoid

![Masked][masked]

Step 5. Draw lines on the lanes by Hough transform

![Lines][lines]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by taking the medians of the slopes and intercepts of the segments from Hough transform.



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ...

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
