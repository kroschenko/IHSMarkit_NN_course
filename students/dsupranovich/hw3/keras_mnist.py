#!/usr/bin/env python

from keras.utils import to_categorical, plot_model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

num_classes = 10

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def nn1_model():
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
					 activation='relu',
					 input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
				  optimizer='Adadelta',
				  metrics=['accuracy'])
	return model

def nn2_model():
	model = Sequential()
	model.add(Conv2D(50, kernel_size=(3, 3),
					 activation='relu',
					 input_shape=input_shape))
	model.add(Conv2D(100, (3, 3), activation='relu'))
	model.add(Conv2D(500, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(1000, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss='categorical_crossentropy',
				  optimizer='Adadelta',
				  metrics=['accuracy'])
	return model

nn_model = nn2_model()

early_stopping =EarlyStopping(monitor='val_acc', patience=10)
bst_model_path = 'convnn_{epoch:02d}-{val_acc:.4f}.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, monitor='val_acc', save_best_only=True)

#plot_model(nn_model, to_file='./model.png', show_shapes=True)

nn_model.fit(x_train, y_train,
		  batch_size=64,
		  epochs=2000,
		  verbose=1,
		  validation_data=(x_test, y_test),
		  callbacks=[early_stopping, model_checkpoint])
score = nn_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
