import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

""" Load Self-Supervised Dataset """
def load_selfsupervised(folderpath, foldername, target_img_size=(100, 100, 3), preprocessing_function=None, batch_size=32):
	"""
	@params		folderpath, foldername, target_img_size, preprocessing_function, batch_size
	@returns	train_data, test_data

	folderpath: path to data directory
	foldername: name of the folder containing the images
	/folderpath
	  /foldername
	  	/img_000000.png
	  	/img_000001.png
	  	/...
	target_img_size: image size needed for the model, eg (100, 100, 3) [default]
	preprocessing_function: function to preprocess the data
	batch_size: define batch size

	Function returns 2 image data generators (train, test) with dataflow from directory
	"""
	# training data
	train_data_gen = ImageDataGenerator(preprocessing_function=preprocessing_function,
	                                    validation_split=0.2)
	train_data = train_data_gen.flow_from_directory(folderpath, classes=[foldername],
	                                                target_size=target_img_size[:2],
	                                                color_mode='rgb', shuffle=True,
	                                                class_mode='input', batch_size=batch_size,
	                                                subset='training')

	# test data
	test_data_gen = ImageDataGenerator(preprocessing_function=preprocessing_function,
	                                   validation_split=0.2)
	test_data = test_data_gen.flow_from_directory(folderpath, classes=[foldername],
	                                                target_size=target_img_size[:2],
	                                                color_mode='rgb', shuffle=True,
	                                                class_mode='input', batch_size=batch_size,
	                                                subset='validation')

	return train_data, test_data 



""" Load binary Dataset """
def load_binary(folderpath, target_img_size=(100, 100, 3)):
	"""
	@params		folderpath, target_img_size
	@returns	train_data, test_data

	folderpath: path where the data is stored with the following structure
	/binarylabels.txt
	/images
	  /anger_000000.png
	  /anger_000001.png
	  /...
	target_img_size: image size needed for the model, eg (100, 100, 3) [default]

	Function returns 2 image data generators (train, test) with dataflow from directory
	"""
	labelling_list = open(folderpath + 'binarylabels.txt', 'r').read().strip().split('\n')

	train_set = {'filename':[], 'class':[]}

	for line in labelling_list:
	    filename, ishappy = line.split(' ')
	    train_set['filename'].append(filename)
	    if int(ishappy):
	        train_set['class'].append('happy')
	    else:
	        train_set['class'].append('nothappy')

	print("Counted %s images in total" % len(train_set['filename']))

	train_set = pd.DataFrame(data=train_set)

	train_data_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, 
	                                    height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
	                                    horizontal_flip=True, validation_split=0.2)
	train_data = train_data_gen.flow_from_dataframe(train_set, directory = folderpath + 'images/', batch_size=32, 
	                                                color_mode='rgb', shuffle=True, class_mode='binary',
	                                                target_size=target_img_size[:2], subset='training')

	test_data_gen = ImageDataGenerator(validation_split=0.2)
	test_data = test_data_gen.flow_from_dataframe(train_set, directory = folderpath + 'images/', batch_size=32, 
	                                              color_mode='rgb', shuffle=True, class_mode='binary',
	                                              target_size=target_img_size[:2], subset='validation')

	return train_data, test_data


""" Load FERplus Dataset """
def load_RAF(folderpath, target_img_size=(100, 100, 3)):
	"""
	@params		folderpath, target_img_size
	@returns	train_data, test_data

	folderpath: path where the data is stored with the following structure
	/labels.txt
	/images
	  /test_0001_aligned.jpg
	  /test_0002_aligned.jpg
	  /test_0003_aligned.jpg
	  /...
	target_img_size: image size needed for the model, eg (100, 100, 3) [default]

	Function returns 2 image data generators (train, test) with dataflow from directory
	"""
	emotion_labels = {  1: 'Surprise',
	                    2: 'Fear',
	                    3: 'Disgust',
	                    4: 'Happiness',
	                    5: 'Sadness',
	                    6: 'Anger',
	                    7: 'Neutral'
	                 }

	labelling_list = open(folderpath + 'labels.txt', 'r').read().strip().split('\n')

	train_set = {'filename':[], 'class':[]}
	test_set = {'filename':[], 'class':[]}

	for line in labelling_list:
	    name, emotion = line.split(' ')
	    name, t = name.split('.')
	    filename = name + '_aligned.' + t
	    if filename.startswith('train'):
	        train_set['filename'].append(filename)
	        train_set['class'].append(emotion_labels[int(emotion)])
	    else :
	        test_set['filename'].append(filename)
	        test_set['class'].append(emotion_labels[int(emotion)])

	print("Counted %s images for traininig" % len(train_set['filename']))
	print("Counted %s images for testing" % len(test_set['filename']))

	train_set = pd.DataFrame(data=train_set)
	test_set = pd.DataFrame(data=test_set)

	train_data_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, 
	                                    height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
	                                    horizontal_flip=True)

	test_data_gen = ImageDataGenerator()

	train_data = train_data_gen.flow_from_dataframe(train_set, directory = folderpath + 'images/', batch_size=32, 
	                                                color_mode='rgb', shuffle=True, class_mode='categorical',
	                                                target_size=target_img_size[:2])

	test_data = test_data_gen.flow_from_dataframe(test_set, directory = folderpath + 'images/', batch_size=32, 
	                                              color_mode='rgb', shuffle=True, class_mode='categorical',
	                                              target_size=target_img_size[:2])

	return train_data, test_data

""" Load FERplus Dataset """
def load_FERplus(folderpath, target_img_size=(100, 100, 3)):
	"""
	@params		folderpath, target_img_size
	@returns	train_data, val_data, test_data

	folderpath: path where the data is stored with the following structure
	/images
	  /fer2013new.csv
	  /FER2013Test
		/fer0032220.png
		/...
	  /FER2013Train
		/fer0000000.png
		/...
	  /FER2013Valid
		/fer0028638.png
		/...
	target_img_size: image size needed for the model, eg (100, 100, 3) [default]

	Function returns 3 image data generators (train, val, test) with dataflow from directory
	"""

	path_fer2013new = folderpath + 'fer2013new.csv'
	train_dir = folderpath + 'images/FER2013Train/'
	test_dir = folderpath + 'images/FER2013Test/'
	val_dir = folderpath + 'images/FER2013Valid/'

	labelling_list = pd.read_csv(path_fer2013new)

	train_set = {'filename':[], 'class':[]}
	test_set = {'filename':[], 'class':[]}
	valid_set = {'filename':[], 'class':[]}

	emotion_labels = {  1: 'neutral',
	                    2: 'happiness',
	                    3: 'surprise',
	                    4: 'sadness',
	                    5: 'anger',
	                    6: 'disgust',
	                    7: 'fear',
	                    8: 'comtempt',
	                    9: 'unknown',
	                    10: 'NF'
	                 }

	for idx, elem in labelling_list.iterrows():

	    # get the vote vector for the emotions
	    vote_vector = elem[2:].to_numpy()
	    # only take the major vote
	    major_vote = np.argmax(vote_vector)
	    emotion = emotion_labels[major_vote+1]  # add 1, as emotion_labels starts with 1


	    # sort out unknown labellings
	    if emotion in ['comtempt', 'unknown', 'NF']:
	        continue

	    # assign to dictionary
	    if elem.Usage == 'Training':
	        train_set['filename'].append(elem['Image name'])
	        train_set['class'].append(emotion)
	    elif elem.Usage == 'PublicTest':
	        valid_set['filename'].append(elem['Image name'])
	        valid_set['class'].append(emotion)
	    elif elem.Usage == 'PrivateTest':
	        test_set['filename'].append(elem['Image name'])
	        test_set['class'].append(emotion)

	print("Counted %s images for traininig" % len(train_set['filename']))
	print("Counted %s images for validation" % len(valid_set['filename']))
	print("Counted %s images for testing" % len(test_set['filename']))

	# make dataframe
	train_set = pd.DataFrame(data=train_set)
	test_set = pd.DataFrame(data=test_set)
	valid_set = pd.DataFrame(data=valid_set)

	# create generators
	# TRAINING
	train_data_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, 
	                                    height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
	                                    horizontal_flip=True)
	train_data = train_data_gen.flow_from_dataframe(train_set, directory=train_dir, batch_size=32, 
	                                                color_mode='rgb', shuffle=True, class_mode='categorical',
	                                                target_img_size=target_img_size[:2])
	# VALIDATION
	val_data_gen = ImageDataGenerator()
	val_data = val_data_gen.flow_from_dataframe(valid_set, directory=val_dir, batch_size=32, 
	                                          color_mode='rgb', shuffle=True, class_mode='categorical',
	                                          target_img_size=target_img_size[:2])
	# TESTING
	test_data_gen = ImageDataGenerator()
	test_data = test_data_gen.flow_from_dataframe(test_set, directory=test_dir, batch_size=32, 
	                                              color_mode='rgb', shuffle=True, class_mode='categorical',
	                                              target_img_size=target_img_size[:2])

	return train_data, val_data, test_data