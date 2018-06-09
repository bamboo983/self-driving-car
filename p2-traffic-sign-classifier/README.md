# **Traffic Sign Recognition**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/train.png "Training Set"
[image2]: ./images/validation.png "Validation Set"
[image3]: ./images/test.png "Test Set"
[image4]: ./traffic-signs-data/web_images/00001.png "Traffic Sign 1"
[image5]: ./traffic-signs-data/web_images/00006.png "Traffic Sign 2"
[image6]: ./traffic-signs-data/web_images/00007.png "Traffic Sign 3"
[image7]: ./traffic-signs-data/web_images/00023.png "Traffic Sign 4"
[image8]: ./traffic-signs-data/web_images/00067.png "Traffic Sign 5"
[image9]: ./traffic-signs-data/web_images/00079.png "Traffic Sign 6"
[image10]: ./traffic-signs-data/web_images/00107.png "Traffic Sign 7"
[image11]: ./traffic-signs-data/web_images/00108.png "Traffic Sign 8"
[image12]: ./images/conv1.png "Convolution Layer 1"
[image13]: ./images/conv1_relu.png "Convolution Layer 1 RELU"
[image14]: ./images/conv1_pool.png "Convolution Layer 1 Max Pool"
[image15]: ./images/conv2.png "Convolution Layer 2"
[image16]: ./images/conv2_relu.png "Convolution Layer 2 RELU"
[image17]: ./images/conv2_pool.png "Convolution Layer 2 Max Pool"
[image18]: ./images/learning.png "Learning Curve"
[image19]: ./images/original.png "Original"
[image20]: ./images/gray.png "Grayscale"

### Data Set Summary & Exploration

#### 1. Dataset Summary

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory Visualization

Here are the exploratory visualization of the data set. They are histograms
showing how training, validation and test sets distributed. We could see that
the three different data sets have similar distributions of the traffic signs.

![train][image1]
![validation][image2]
![test][image3]

### Design and Test a Model Architecture

#### 1. Data Preprocessing

First, I decided to convert the images to grayscale because it decreased the
depth of image from 3 to 1, which accelerated the training speed dramatically.
In addition, based on [Yann LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), colorless
recognition was not less accurate.

Here is an example of a traffic sign image before and after grayscaling.

![original][image19] ![gray][image20]

Second, I normalized the image data symmetrically because it could achieve
consistency in dynamic range of the dataset and improved the accuracy.

#### 2. Model Architecture

My final model consisted of the following layers:

| Layer         	| Description                                      |
|:---------------:|:------------------------------------------------:|
| Input         	| 32x32x1 grayscale image                          |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x6       |
| RELU            |	                                                 |
| Max pooling	    | 2x2 stride, outputs 14x14x6                      |
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16      |
| RELU            |                                                  |
| Max pooling     | 2x2 stride, outputs 5x5x16                       |
| Fully connected | Multi-scale convolutional networks<br> concatenation, outputs 1576 |
| Fully connected | outputs 120                                      |
| RELU            |                                                  |
| Dropout         | 0.5 dropout rate, outputs 120                    |
| Fully connected | outputs 84                                       |
| RELU            |                                                  |
| Dropout         | 0.5 dropout rate, outputs 84                     |
| Fully connected | outputs 43                                       |
| Softmax         | The classification probabilities       					 |


#### 3. Model Training

To train the model, I used LeNet with multi-scale convolutional networks. The
hyperparameters were listed below.
* Adam optimizer
* Batch size of 128
* Epochs of 20
* Learning rate of 0.001

Adam was an adaptive moment estimation, which combines the advantages of AdaGrad
(Adaptive Gradient Algorithm) and RMSProp (Root Mean Square Propagation). I did
not change the default batch size and learning rate, and 20 epochs was sufficient
to meet the minimum requirement of validation set accuracy of 93%.

#### 4. Solution Approach

My final model results were:
* Training set accuracy of 99.6%
* Validation set accuracy of 95.7%
* Test set accuracy of 93.6%

I chose LeNet as the architecture of convolutional neural networks because I was
familiar with it from the lecture, and it could achieve the accuracy of 99%
according to the [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).
With LeNet solution from the lecture on colored images, I could only achieve
validation set accuracy of 89%, as the instructions said.

Then, I normalized the dataset by subtracting the mean of whole dataset and
dividing by the difference of the maximum and minimum value of it. To avoid
overfitting, dropout was applied on the fully connected layers. As a result, the
validation set accuracy increased to 92%.

To further improve the accuracy to meet the project requirement, I read the
paper and knew that multi-scale neural networks improved the accuracy and the
colored images did not help much, so two convolutional layers were concatenated
and grayscale images were used to train the model. Finally, I could easily get
the accuracy of 95% within only 20 epochs.

![learning][image18]

As the learning curve shown, the training error was 99.6% and the validation
error was 95.7%. There was no sign of overfitting, or the validation error curve
would increased at some point. After applied on the test set, the accuracy of
93.6% proved this model worked well on German traffic signs.

### Test a Model on New Images

#### 1. Acquiring New Images

Here are eight German traffic signs that I found on the [web](http://benchmark.ini.rub.de/Dataset/GTSRB_Online-Test-Images.zip):

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11]

I picked some dark images on purpose. The 5th and 7th images might be difficult
to classify because they were almost black, which hardly recognized by humans.
In the beginning, I thought these two images were crashed to all black. After
zooming the size and adjusting the brightness by image processing software, I
just realized that there were really traffic signs in the images.

#### 2. Performance on New Images

Here are the results of the prediction:

| Image                                        | Prediction                                   |
|:--------------------------------------------:|:--------------------------------------------:|
| Speed limit (60km/h)                         | Speed limit (60km/h)                         |
| No passing for vehicles over 3.5 metric tons | No passing for vehicles over 3.5 metric tons |
| Dangerous curve to the right                 | Dangerous curve to the right                 |
| Right-of-way at the next intersection        | Right-of-way at the next intersection        |
| General caution                              | General caution                              |
| Speed limit (120km/h)                        | Speed limit (120km/h)                        |
| No passing for vehicles over 3.5 metric tons | No passing for vehicles over 3.5 metric tons |
| Keep right                                   | Keep right                                   |


The model was able to correctly guess 8 of the 8 traffic signs, which gives an
accuracy of 100%. This compares favorably to the accuracy on the test set of
93.6%. Therefore, this accuracy should be a reasonable result with small amount
of images.

#### 3. Model Certainty - Softmax Probabilities

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook.

For the first image, the model was very sure that the image was a speed limit
(60km/h), which was a correct classification.

| Probability | Prediction           |
|:-----------:|:--------------------:|
| .99919957   | Speed limit (60km/h) |
| .00078760   | Speed limit (80km/h) |
| .00001199   | Ahead only           |
| .00000055   | Speed limit (50km/h) |
| .00000018   | Speed limit (30km/h) |

For the second image, the model was very sure that the image was a no passing
for vehicles over 3.5 metric tons, which was a correct classification.

| Probability | Prediction           |
|:-----------:|:--------------------:|
| .99989140   | No passing for vehicles over 3.5 metric tons |
| .00004567   | Speed limit (100km/h)                        |
| .00004526   | Speed limit (80km/h)                         |
| .00001751   | Speed limit (120km/h)                        |
| .00000008   | Vehicles over 3.5 metric tons prohibited     |

For the third image, the model had high confidence, over 90% probability, that
the image was a no passing for vehicles over 3.5 metric tons, which was a
correct classification.

| Probability | Prediction                   |
|:-----------:|:----------------------------:|
| .92530441   | Dangerous curve to the right |
| .03957299   | Keep right                   |
| .01007588   | Slippery road                |
| .00708711   | Turn left ahead              |
| .00477633   | Bicycles crossing            |

For the fourth image, the model was surprisingly low confident of its guess.
Although the answer was correct, the probability was less than 50%. High
contrast between the traffic sign and background probably affected the model.

| Probability | Prediction                            |
|:-----------:|:-------------------------------------:|
| .46520102   | Right-of-way at the next intersection |
| .33832934   | Beware of ice/snow                    |
| .04714746   | General caution                       |
| .02924148   | Double curve                          |
| .02525485   | Pedestrians                           |

For the fifth image, the model had correct guess of the image with low
brightness. However, we could see that the brightness did affect the probability
of the classification, which was below 90%.

| Probability | Prediction                |
|:-----------:|:-------------------------:|
| .82382184   | General caution           |
| .09473898   | Traffic signals           |
| .01744064   | Pedestrians               |
| .01436838   | Road work                 |
| .00918836   | Road narrows on the right |

For the sixth image, the model was very sure that the image was a speed limit
(120km/h), which was a correct classification.

| Probability | Prediction            |
|:-----------:|:---------------------:|
| .97637284   | Speed limit (120km/h) |
| .01744896   | Speed limit (100km/h) |
| .00386448   | Speed limit (70km/h)  |
| .00115654   | Roundabout mandatory  |
| .00053238   | Speed limit (80km/h)  |

For the seventh image, the model had correct guess of the image with low
brightness. However, we could see that the brightness did affect the probability
of the classification, which was below 90%.

| Probability | Prediction                                   |
|:-----------:|:--------------------------------------------:|
| .85956758   | No passing for vehicles over 3.5 metric tons |
| .06900898   | Priority road                                |
| .01941952   | Speed limit (80km/h)                         |
| .01237051   | No passing                                   |
| .01029186   | Speed limit (50km/h)                         |

For the eighth image, the model was very sure that the image was a keep right,
which was a correct classification.

| Probability | Prediction                   |
|:-----------:|:----------------------------:|
| .99990392   | Keep right                   |
| .00004321   | Dangerous curve to the right |
| .00003829   | Turn left ahead              |
| .00000575   | Priority road                |
| .00000379   | Stop                         |

### Visualizing the Neural Network
#### 1. Feature Maps

* Convolutional Layer 1: Detected the shape of the traffic signs and numbers.

![Conv1][image12]

* Layer 1 RELU: Blacken the pixels which brightness were below the threshold.

![Conv1 RELU][image13]

* Layer 1 Max Pooling: Picked the maximum value in the square filter to mitigate
the motion blur.

![Conv1 Pool][image14]

* Convolutional Layer 2: Separated different parts of the previous layer output.

![Conv2][image15]

* Layer 2 RELU: Blacken the pixels which brightness were below the threshold.

![Conv2 RELU][image16]

* Layer 2 Max Pooling: Picked the maximum value in the square filter to mitigate
the motion blur.

![Conv2 Pool][image17]
