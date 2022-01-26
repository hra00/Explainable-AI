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

    def get_CAM(self, images):
        # prepare images
        image_batch = []
        if isinstance(images, list):
            for image in images:
                x = cv2.resize(image, self.img_shape[:2])
                x = self.preprocess(x)
                image_batch.append(x)
        else:
            x = cv2.resize(images, self.img_shape[:2])
            x = self.preprocess(x)
            image_batch.append(x)
        image_batch = np.array(image_batch)

        # predict
        model_out, feature_maps = self.model_CAM.predict(image_batch)
        
        # make CAMs
        weights = self.model.layers[-1].weights[0]
        # for each image repeat
        CAMs = []
        for idx in range(image_batch.shape[0]):
            max_idx = np.argmax( np.squeeze(model_out[idx]) )
            print(f"Maximum index {max_idx} with confidence {model_out[idx][max_idx]*100:.2f}%")

            winning_weights = weights[:, max_idx]
            CAM = np.sum(feature_maps[idx] * winning_weights, axis=2)

            CAMs.append(CAM)
        CAMs = np.array(CAMs)
        
        heatmaps = []
        for idx, img in enumerate(images):
            heatmap = cv2.resize(CAM, self.img_shape[:2])
            heatmaps.append(heatmap)
        heatmaps = np.array(heatmaps)

        return heatmaps, model_out


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
	model.load_weights('./models/' + model_name)

	# last convolutional name
	LAST_CONV_NAME = 'block3_conv3'

	
	if show_image_batch:
		# multiple images
		image_names = ['RAF/test_0266_aligned.jpg', 'RAF/test_0001_aligned.jpg', 'happy.png']
		images = []
		for name in image_names:
		    img = cv2.imread('./data/' + name)
		    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		    images.append(img)
	else:
		# single image
		images = img = cv2.imread('./data/happy.png')
		images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

	
	# Visualizer ---------------
	vis = Visualizer(model, IMG_SHAPE, LAST_CONV_NAME, preprocess)
	
	heatmaps, preds = vis.get_CAM(images)
	# --------------------------


	# visualize results
	if isinstance(images, list):
	    plt.subplots(1, len(images), figsize=(14,6))
	    for idx, image in enumerate(images):
	        x = cv2.resize(image, IMG_SHAPE[:2])
	        plt.subplot(1, len(images), idx+1)
	        plt.imshow(x, alpha=0.5)
	        plt.imshow(heatmaps[idx], cmap='jet', alpha=0.5)
	        plt.title(f'{vec2label(preds[idx])}: {np.max(preds[idx])*100:.2f}%')
	        plt.axis('off')
	else:
	    plt.figure(figsize=(10,10))
	    x = cv2.resize(images, IMG_SHAPE[:2])
	    plt.imshow(x, alpha=0.5)
	    plt.imshow(heatmaps[0], cmap='jet', alpha=0.5)
	    plt.title(f'{vec2label(preds[0])}: {np.max(preds[0])*100:.2f}%')
	    plt.axis('off')
	    
	plt.show()

