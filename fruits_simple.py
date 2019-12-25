import cv2
import keras
import os
import numpy as np
from time import time
from sys import argv
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

                     
def parse_args():
    epochs = 10
    batch = 256
    close = 1700
    return epochs, batch, close


def load(path):
	images = []
	labels = []
	classes = os.listdir(path)
	for i in range(120):
		cur_path = path + classes[i] + '/'
		fruits = os.listdir(cur_path)
		for x in fruits:
			image = cv2.imread(cur_path + x)
			image = cv2.resize(image, (50, 50))
			images.append(image)
			labels.append(i)
	tensor = np.array(images)
	return tensor, labels


def main():
	start = time()
	epochs, batch, close = parse_args()
	train_images, train_labels = load('fruits-360_dataset/fruits-360/Training/')
	test_images, test_labels = load('fruits-360_dataset/fruits-360/Test/')     

	print('LOAD ' + str(time() - start))
	
	in_shape = train_images[0].shape

	train_images = train_images.astype('float32') / 255
	train_labels = to_categorical(train_labels)
	
	test_images = test_images.astype('float32') / 255
	test_labels = to_categorical(test_labels)
	
	num_classes = 120

	model = Sequential()
	
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	
	model.add(Flatten())
	model.add(Dense(close, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	
	model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                
	start = time()
	model.fit(train_images, train_labels, epochs=epochs, batch_size=batch)
	train_time = time() - start

	test_loss, test_acc = model.evaluate(test_images, test_labels)
	print('\nTest accuracy:', test_acc)
	print('Time: ', train_time)

	model.summary()


if __name__ == '__main__':
	main()
