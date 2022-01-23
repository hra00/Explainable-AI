from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Activation, GlobalAveragePooling2D
import tensorflow as tf

class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, kernel_num, kernel_size, strides=1, padding='same', kernel_initializer='he_normal', name=None):
        super(ConvLayer, self).__init__(name=name)
        
        # convolutional layers
        self.conv = Conv2D(kernel_num,
        	kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            activation=None)
        # batch normalization (before activation!)
        self.batchnorm = BatchNormalization()
        # activation
        self.activation = Activation("relu")
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x


def get_base_model2(image_shape):
	
	model = tf.keras.Sequential()

	# input layer
	model.add(tf.keras.layers.InputLayer(input_shape=image_shape))

	# block 1 - 64 filters (2 times)
	model.add(ConvLayer(64, kernel_size=(3,3), padding="same", kernel_initializer="he_normal", name="block1_conv1"))
	model.add(ConvLayer(64, kernel_size=(3,3), padding="same", kernel_initializer="he_normal", name="block1_conv2"))
	model.add(MaxPool2D(3,3, name="maxpool_1"))

	# block 2 - 96 filters (3 times)
	model.add(ConvLayer(96, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', name="block2_conv1"))
	model.add(ConvLayer(96, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', name="block2_conv2"))
	model.add(ConvLayer(96, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', name="block2_conv3"))
	model.add(MaxPool2D(3,3, name="maxpool_2"))

	# block 3 - 128 filters (3 times)
	model.add(ConvLayer(128, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', name="block3_conv1"))
	model.add(ConvLayer(128, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', name="block3_conv2"))
	model.add(ConvLayer(128, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', name="block3_conv3"))

	# global average pooling layer
	model.add(GlobalAveragePooling2D(name='GAP'))

	return model


def get_base_model(image_shape):
	"""
	@param		size of the image to build the model
	@return		base model consisting of 3 convolutional blocks
				followed by global average pooling
	"""
	model = tf.keras.Sequential()

	# block 1 - 64 filters (2 times)
	model.add(Conv2D(64, kernel_size=(3,3), activation=None, padding='same', input_shape=image_shape, kernel_initializer='he_normal'))
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