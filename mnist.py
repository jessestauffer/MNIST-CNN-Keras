import numpy as np
from random import randint
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

def generator(features, labels, batch_size):

	# create empty arrays to contain batch of features and labels
	batch_features = np.zeros((batch_size, 28, 28, 1))
	batch_labels = np.zeros((batch_size, 10))

	while True:
		for i in range(batch_size):
			# choose random index in features
			index = np.random.randint(0, len(features)-1)
			random_augmented_image, random_augmented_label = features[index], labels[index]
			batch_features[i] = random_augmented_image
			batch_labels[i] = random_augmented_label

		yield batch_features, batch_labels

# load pre-shuffled MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# pre-process data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
print(Y_train.shape)

# build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(generator(X_train, Y_train, 32), steps_per_epoch=X_train.shape[0] / 32, epochs=10, verbose=1)