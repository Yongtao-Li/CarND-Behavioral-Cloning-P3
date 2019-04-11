import csv
import cv2
import numpy as np

### read the data log file
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)
        
### read all available images for features and steering command for lables
### convert BGR to RGB image
images = []
measurements = []
correction = 0.1
for line in lines:
    ### center image
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    imageBGR = cv2.imread(current_path)
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
    images.append(imageRGB)
    measurement = float(line[3])
    measurements.append(measurement)
    
    ### left image
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    imageBGR = cv2.imread(current_path)
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
    images.append(imageRGB)
    measurement = float(line[3]) + correction
    measurements.append(measurement)
    
    ### right image
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    imageBGR = cv2.imread(current_path)
    imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
    images.append(imageRGB)
    measurement = float(line[3]) - correction
    measurements.append(measurement)

### set up the Nvidia neural network model
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPool2D
import matplotlib.pyplot as plt

### train and validate the model

model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))

model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))

model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))

model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())

model.add(Dense(100))

model.add(Dense(50))

model.add(Dense(10))

model.add(Dense(1))

model.summary()

model.compile(loss = 'mse', optimizer = 'adam')

history_object = model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 7, verbose = 1)

### print the keys contained in the history object
print (history_object.history.keys())

### visualize training and validation loss
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

### save the trained model
model.save('model_iteration03.h5')
        
    
    
