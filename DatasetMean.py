import numpy as np
import cv2
import os
import tensorflow as tf
import time

def get_mean_RAF():
	DATA_DIR = './data/RAF/images'

	t0 = time.time()

	X_train = []

	for filename in os.listdir(DATA_DIR):
		if not filename.startswith('train'):
			continue

		fullpath = os.path.join(DATA_DIR, filename)
		X_train.append(
	            tf.keras.preprocessing.image.img_to_array(
	                tf.keras.utils.load_img(
	                    fullpath, grayscale=False, color_mode='rgb',
	                    #target_size=IMG_SIZE, interpolation='nearest'
		)))

	X_train = np.array(X_train, dtype='double')			# caution: dtype must be more precise than float32!!!
	print(X_train.shape)

	print("Gathered %s images in %.5f seconds" % (X_train.shape[0], time.time() - t0))


	over_axis_mean = np.mean(X_train, axis=(0, 1, 2))
	print("Mean over axis: ", over_axis_mean)
	print("Mean per channel\n0: %.4f\n1: %.4f\n2: %.4f" % (
			np.mean(X_train[..., 0]),
			np.mean(X_train[..., 1]),
			np.mean(X_train[..., 2])
		))

	over_axis_std = np.std(X_train, axis=(0, 1, 2))
	print("Std over axis: ", over_axis_std)
	print("Std per channel\n0: %.4f\n1: %.4f\n2: %.4f" % (
			np.std(X_train[..., 0]),
			np.std(X_train[..., 1]),
			np.std(X_train[..., 2])
		))

	print('Gathered images and calculated statistics in %.4f' % (time.time() - t0))


def get_mean_FERplus():
	TRAIN_DIR = './data/ferplus2013/images/FER2013Train'

	t0 = time.time()

	X_train = []
	for filename in os.listdir(TRAIN_DIR):
		fullpath = os.path.join(TRAIN_DIR, filename)
		X_train.append(
	            tf.keras.preprocessing.image.img_to_array(
	                tf.keras.utils.load_img(
	                    fullpath, grayscale=False, color_mode='rgb',
	                    #target_size=IMG_SIZE, interpolation='nearest'
		)))

	X_train = np.array(X_train, dtype='double')			# caution: dtype must be more precise than float32!!!
	print(X_train.shape)

	print("Gathered %s images in %.5f seconds" % (X_train.shape[0], time.time() - t0))


	over_axis_mean = np.mean(X_train, axis=(0, 1, 2))
	print("Mean over axis: ", over_axis_mean)
	print("Mean per channel\n0: %.4f\n1: %.4f\n2: %.4f" % (
			np.mean(X_train[..., 0]),
			np.mean(X_train[..., 1]),
			np.mean(X_train[..., 2])
		))

	over_axis_std = np.std(X_train, axis=(0, 1, 2))
	print("Std over axis: ", over_axis_std)
	print("Std per channel\n0: %.4f\n1: %.4f\n2: %.4f" % (
			np.std(X_train[..., 0]),
			np.std(X_train[..., 1]),
			np.std(X_train[..., 2])
		))

	print('Gathered images and calculated statistics in %.4f' % (time.time() - t0))


if __name__ == "__main__":
	#get_mean_FERplus()
	get_mean_RAF()