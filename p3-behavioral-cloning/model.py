import csv
import cv2
import numpy as np

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
angles = []
for line in lines[1:]:
    # Get image source path
    center_source_path = line[0]
    left_source_path = line[1].lstrip(' ')
    right_source_path = line[2].lstrip(' ')

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
    center_angle = float(line[3])
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

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
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
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1)
model.save('model.h5')

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model Mean Squared Error Loss')
plt.ylabel('Mean Squared Error Loss')
plt.xlabel('Epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
