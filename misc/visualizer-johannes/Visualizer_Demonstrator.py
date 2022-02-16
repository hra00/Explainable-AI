import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from ModelGenerator import get_base_model2

class Visualizer:
	def __init__(self, model, img_shape, last_conv_name, preprocessing_fn):
		self.model = model
		self.img_shape = img_shape
		self.conv_name = last_conv_name
		self.preprocess = preprocessing_fn

		self.last_conv_layer = self.model.get_layer(self.conv_name)

		self.model_CAM = tf.keras.Model(inputs=model.inputs, outputs=[model.output, self.last_conv_layer.output])
		
		self.weights = self.model.layers[-1].weights[0]


	def get_CAM(self, images):
		# prepare images
		x = cv2.resize(images, self.img_shape[:2])
		x = self.preprocess(x)
		x = np.expand_dims(x, axis=0)

		# predict
		model_out, feature_maps = self.model_CAM.predict(x)

		# get index of predicted class
		max_idx = np.argmax( np.squeeze(model_out) )

		# get winning weights
		winning_weights = self.weights[:, max_idx]


		# create CAM
		CAM = np.sum( np.squeeze(feature_maps) * winning_weights, axis=2)

		# create heatmap
		heatmap = cv2.resize(CAM, self.img_shape[:2])

		return heatmap, model_out




""" Demonstration """
def preprocess(x):	
	mean = [129.4432, 129.4432, 129.4432]
	std = [64.87448751, 64.87448751, 64.87448751]
	# ensure image format
	x = np.array(x, dtype='float32')
    
    # normalize
	x[..., 0] -= mean[0]
	x[..., 1] -= mean[1]
	x[..., 2] -= mean[2]
	if std is not None:
		x[..., 0] /= std[0]
		x[..., 1] /= std[1]
		x[..., 2] /= std[2] 
	return x

emotion_labels = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
def vec2label(onehot_vec):
	major_vote = np.argmax(onehot_vec)
	return emotion_labels[major_vote]


if __name__ == "__main__":

	show_image_batch = False

	# image shape
	IMG_SHAPE = (100, 100, 3)

	# model
	model_name = 'FERplus-impr-std_0124-1040_weights.h5'
	model = get_base_model2(IMG_SHAPE)
	model.add(tf.keras.layers.Dense(7, activation='softmax', name="softmax"))
	model.load_weights(model_name)

	# last convolutional name
	LAST_CONV_NAME = 'block3_conv3'

	img = cv2.imread('./happy.png')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	
	# Visualizer ---------------
	vis = Visualizer(model, IMG_SHAPE, LAST_CONV_NAME, preprocess)
	
	heatmap, preds = vis.get_CAM(img)
	# --------------------------


	# visualize results	
	plt.figure(figsize=(10,10))
	x = cv2.resize(img, IMG_SHAPE[:2])
	plt.imshow(x, alpha=0.5)
	plt.imshow(heatmap, cmap='jet', alpha=0.5)
	plt.title(f'{vec2label(preds[0])}: {np.max(preds[0])*100:.2f}%')
	plt.axis('off')
	    
	plt.show()

