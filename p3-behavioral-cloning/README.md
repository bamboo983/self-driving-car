# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of five convolution neural networks. Three of them are 5x5
filter sizes and depths from 24, 36, 128, and the other two are 3x3 filter
sizes and depth 64.

The input 160x320 RGB image is normalized with Keras lambda layer. To mask
useless part of image, a cropping layer is used to trim 70 pixels from the top
and 25 pixels from the bottom. The model includes RELU layers to introduce
nonlinearity and dropout layers between fully-connected layers to avoid
overfitting.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers with 50% drop rate in order to reduce overfitting.
It was tested by running it through the simulator and ensuring that the vehicle
could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a
combination of center lane driving, left and right cameras steering angles
corrected from the center, flipped images of center, left and right cameras.
That is to say, an image of one position could produce six samples of it.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the sample
training data provided by project resources to build the model and see how it
performed.

My first step was to use a convolution neural network model referenced from
Nvidia. I thought this model might be appropriate because it was more powerful
than LeNet. This model was trained with center images only and performed terribly.
The car drove away from the track easily and never came back again.

To learn how to turn back to the center, the images of left and right cameras
were used. By adding these new training data, the car drove more smoothly but
stuck after a long straight bridge. It decided to turn left easily because the
training set was left turn only.

Then, I slipped all the samples horizontally to balance the training set. It
turned out to work nicely. The car could almost drive at the center. At the end
of the process, the vehicle is able to drive autonomously around the track one
without leaving the road.

#### 2. Final Model Architecture

| Layer         	| Description                                      |
|:---------------:|:------------------------------------------------:|
| Input         	| 160x320x3 image with normalization               |
| Cropping        | 70 pixels from top and 25 pixels from bottom, outputs 65x320x3 |
| Convolution 5x5 |	2x2 stride, valid padding, outputs 31x158x24     |
| RELU      	    |                                                  |
| Convolution 5x5 |	2x2 stride, valid padding, outputs 14x77x36      |
| RELU      	    |                                                  |
| Convolution 5x5 |	2x2 stride, valid padding, outputs 5x37x48       |
| RELU      	    |                                                  |
| Convolution 3x3 |	1x1 stride, valid padding, outputs 3x35x64       |
| RELU      	    |                                                  |
| Convolution 3x3 |	1x1 stride, valid padding, outputs 1x33x64       |
| RELU      	    |                                                  |
| Flatten         | outputs 2112                                     |
| Fully connected | outputs 100                                      |
| Dropout         | 0.5 dropout rate, outputs 100                    |
| Fully connected | outputs 50                                       |
| Dropout         | 0.5 dropout rate, outputs 50                     |
| Fully connected | outputs 1                                        |

#### 3. Creation of the Training Set & Training Process

I used the sample training data of track one to train and validate the model.
It turned out to be work nicely. Then, I would like to train the car to drive
successfully on track two and found it was very difficult to generalize
the model.

First, the road was up and down, left and right consecutively. Even controlling
by hands, I had to drive slowly and carefully to finish a lap. The throttle
should be increased when the car drove up then decreased when it drove down.
After a while, I finally finished one lap on the right side track like we drove
in the real world.

When I added the new dataset to train the model, the car drove at the right side
of the center on track one and became stuck at the turn easily. Therefore, I
switch back to the original model and can only finished the track one.
