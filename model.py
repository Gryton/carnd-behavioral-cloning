import csv
import cv2
import numpy as np
import random
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, Activation
from keras.layers.pooling import MaxPooling2D
import sklearn
from sklearn.model_selection import train_test_split

main_source = './train_data'

with open(main_source + '/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    lines = [line for line in reader]

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = main_source+'/IMG/'+os.path.basename(batch_sample[0])
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                left_image = cv2.imread(main_source+'/IMG/'+os.path.basename(batch_sample[1]))
                right_image = cv2.imread(main_source+'/IMG/'+os.path.basename(batch_sample[2]))
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                images.append(cv2.flip(center_image, 1))
                angles.append(center_angle)
                angles.append(center_angle+0.3)
                angles.append(center_angle-0.3)
                angles.append(-1.0*center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 85, 320  # Trimmed image format


model = Sequential()
# normalize data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
# crop image to leave road
model.add(Cropping2D(cropping=((55, 20), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 4*len(train_samples), validation_data=validation_generator,
        nb_val_samples=4*len(validation_samples), nb_epoch=3, verbose=1)
model.save('model.h5')
