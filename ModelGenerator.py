from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Activation, GlobalAveragePooling2D
import tensorflow as tf

def get_base_model(image_size):
	"""
	@param		size of the image to build the model
	@return		base model consisting of 3 convolutional blocks
				followed by global average pooling
	"""
	model = tf.keras.Sequential()

	# block 1 - 64 filters (2 times)
	model.add(Conv2D(64, kernel_size=(3,3), activation=None, padding='same', input_shape=image_size, kernel_initializer='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(64, kernel_size=(3,3), activation=None, padding='same', kernel_initializer='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(MaxPool2D(3,3))

	# block 2 - 96 filters (3 times)
	model.add(Conv2D(96, kernel_size=(3,3), activation=None, padding='same', kernel_initializer='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(96, kernel_size=(3,3), activation=None, padding='same', kernel_initializer='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(96, kernel_size=(3,3), activation=None, padding='same', kernel_initializer='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	model.add(MaxPool2D(3,3))

	# block 3 - 128 filters (3 times)
	model.add(Conv2D(128, kernel_size=(3,3), activation=None, padding='same', kernel_initializer='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(128, kernel_size=(3,3), activation=None, padding='same', kernel_initializer='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Conv2D(128, kernel_size=(3,3), activation=None, padding='same', kernel_initializer='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation('relu', name='last_conv_out'))

	# global average pooling over feature maps
	model.add(GlobalAveragePooling2D())

	return model

if __name__ == '__main__':
	# define image size
	IMG_SIZE = (100, 100, 3)

	# create model
	mymodel = get_base_model( IMG_SIZE )

	# add classification layer
	mymodel.add(tf.keras.layers.Dense(10, activation='softmax'))

	# print summary
	print( mymodel.summary() )