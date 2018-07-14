import csv

def load_samples(filename, samples):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        lines = []
        for line in reader:
            samples.append(line)
        return samples

samples = []
samples = load_samples('./data/1/driving_log.csv', samples)
# samples = load_samples('./data/2/driving_log.csv', samples)
samples = samples[1:]
print('Number of samples: ', str(len(samples) * 6))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # Get image source path
                center_source_path = batch_sample[0]
                left_source_path = batch_sample[1].lstrip(' ')
                right_source_path = batch_sample[2].lstrip(' ')

                # Add image path prefix
                center_image_path = './data/' + center_source_path
                left_image_path = './data/' + left_source_path
                right_image_path = './data/' + right_source_path

                center_image = cv2.imread(center_image_path)
                left_image = cv2.imread(left_image_path)
                right_image = cv2.imread(right_image_path)

                center_image_flipped = np.fliplr(center_image)
                left_image_flipped = np.fliplr(left_image)
                right_image_flipped = np.fliplr(right_image)

                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                images.append(center_image_flipped)
                images.append(left_image_flipped)
                images.append(right_image_flipped)

                correction = 0.2
                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                center_angle_flipped = -center_angle
                left_angle_flipped = -left_angle
                right_angle_flipped = -right_angle

                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
                angles.append(center_angle_flipped)
                angles.append(left_angle_flipped)
                angles.append(right_angle_flipped)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D, Dropout
import matplotlib.pyplot as plt

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
        len(train_samples)*6, validation_data=validation_generator, \
        nb_val_samples=len(validation_samples)*6, nb_epoch=2)
model.save('model.h5')

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model Mean Squared Error Loss')
plt.ylabel('Mean Squared Error Loss')
plt.xlabel('Epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
