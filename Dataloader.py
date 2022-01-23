from pathlib import Path
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

# https://www.pyimagesearch.com/2021/06/14/a-gentle-introduction-to-tf-data-with-tensorflow/


""" Load RAF-DB Dataset with target vector """
def load_RAF_targetvector(folderpath, target_img_size=(100, 100), preprocessing_function=None, batch_size=32):
    """
    @params		folderpath, target_img_size
    @returns	train_data, test_data

    folderpath: path where the data is stored with the following structure
    /distribution_basic.txt
    /images
      /test_0001_aligned.jpg
      /test_0002_aligned.jpg
      /test_0003_aligned.jpg
      /...
    target_img_size:	 	image size needed for the model, eg (100, 100) [default]
    batch_size:		 		batch size for the dataset
    preprocessing_function:	preprocessing for the image

    Function returns train_data, val_data, test_data ImageDataGenerators
    """
    # get file and folder paths
    labelling_list = open(folderpath + 'distribution_basic.txt', 'r').read().strip().split(' \n')
    img_dir = folderpath + 'images/'

    X_train = []; Y_train = []
    X_test = []; Y_test = []
    Y_test_oneclass = [];
    
    for name_vector_string in labelling_list:
        # separate line of labelling list into image name and target vector
        splitted_name_vector_string = name_vector_string.split(' ')
        image_name = splitted_name_vector_string[0]
        # typecast target vector from string to float
        target_vector = np.asarray(splitted_name_vector_string[1:], dtype="float32")
        
        # get class label to stratify training / validation split
        max_label = np.argmax(target_vector)

        # add aligned to image
        name, fileend = image_name.split('.')
        filename = name + '_aligned.' + fileend

        # load image
        img = cv2.imread(img_dir + filename)
        if img is None:
            print("Error finding image", filename)
            continue
        # change color channel order (RGB instead of BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # resize image
        if img.shape[:2] != target_img_size:
            img = cv2.resize(img, dsize=target_img_size, interpolation=cv2.INTER_CUBIC)

        if filename.startswith('train'):
            X_train.append(img)
            Y_train.append(target_vector)
        else:
            X_test.append(img)
            Y_test.append(target_vector)
            Y_test_oneclass.append(max_label)

    X_train = np.array(X_train); Y_train = np.array(Y_train)
    X_test = np.array(X_test); Y_test = np.array(Y_test)

    print("Splitting testing dataset into stratified validation and training set")
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, stratify=Y_test_oneclass)


    print("Training\n-", X_train.shape, "\n-", Y_train.shape)
    print("Validation\n-", X_val.shape, "\n-", Y_val.shape)
    print("Testing\n-", X_test.shape, "\n-", Y_test.shape)

    train_data_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, 
                                        height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
                                        horizontal_flip=True,
                                        preprocessing_function=preprocessing_function)
    train_data = train_data_gen.flow(
        x=X_train, y=Y_train,
        batch_size=batch_size,
        shuffle=True)

    val_data_gen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    val_data = val_data_gen.flow(
        x=X_val, y=Y_val,
        batch_size=batch_size,
        shuffle=True)

    test_data_gen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    test_data = test_data_gen.flow(
        x=X_test, y=Y_test,
        batch_size=batch_size,
        shuffle=True)

    return train_data, val_data, test_data


""" Load Fer+ Dataset with target vector """
def label_transform(y):
    return y / np.sum(y)
    
def load_FERplus_targetvector(folderpath, target_img_size=(100, 100), preprocessing_function=None, batch_size=32):
    """
    @params		folderpath, batch_size, target_img_size, preprocessing_function
    @returns	(X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

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
    target_img_size:	 	image size needed for the model, eg (100, 100) [default]
    batch_size:		 		batch size for the dataset
    preprocessing_function:	preprocessing for the image

    Function returns train_data, val_data, test_data ImageDataGenerators
    """

    # get file and folder paths
    path_fer2013new = folderpath + 'fer2013new.csv'
    train_dir = folderpath + 'images/FER2013Train/'
    test_dir = folderpath + 'images/FER2013Test/'
    val_dir = folderpath + 'images/FER2013Valid/'

    # read labelling list
    labelling_list = pd.read_csv(path_fer2013new)

    # array for the data
    X_train = []; Y_train = []
    X_val = []; Y_val = []
    X_test = []; Y_test = []

    # iterate through files and load them
    for idx, elem in labelling_list.iterrows():
        image_name = elem['Image name']
        
        if not isinstance(image_name, str): continue
            
        # get the vote vector for the emotions
        vote_vector = np.array(elem[2:-3].to_numpy(), dtype='float32')
        
        # skip if sum of vote vector is smaller than 1 (less than one vote for image)
        if np.sum(vote_vector) < 1: continue
        
        # transform to probability distribution with label_transform
        prob_dist_vote_vector = vote_vector / np.sum(vote_vector)
        
        if elem.Usage == 'Training':
            # load image
            img = cv2.imread(train_dir + image_name)
            if img is None: continue

            # change color channel order (RGB instead of BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # resize image
            if img.shape[:2] != target_img_size:
                img = cv2.resize(img, dsize=target_img_size, interpolation=cv2.INTER_CUBIC)

            # fill arrays
            X_train.append(img)
            Y_train.append(prob_dist_vote_vector)
            
        elif elem.Usage == 'PrivateTest':
            # load image
            img = cv2.imread(test_dir + image_name)
            if img is None: continue

            # change color channel order (RGB instead of BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # resize image
            if img.shape[:2] != target_img_size:
                img = cv2.resize(img, dsize=target_img_size, interpolation=cv2.INTER_CUBIC)

            # fill arrays
            X_val.append(img)
            Y_val.append(prob_dist_vote_vector)
            
        elif elem.Usage == 'PublicTest':
            # load image
            img = cv2.imread(val_dir + image_name)
            if img is None: continue

            # change color channel order (RGB instead of BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # resize image
            if img.shape[:2] != target_img_size:
                img = cv2.resize(img, dsize=target_img_size, interpolation=cv2.INTER_CUBIC)

            # preprocess image
            #if preprocessing_function is not None:
            #    img = preprocessing_function(img)

            # fill arrays
            X_test.append(img)
            Y_test.append(prob_dist_vote_vector)
            
    X_train = np.array(X_train); Y_train = np.array(Y_train)
    X_val = np.array(X_val); Y_val = np.array(Y_val)
    X_test = np.array(X_test); Y_test = np.array(Y_test)

    print("Training\n-", X_train.shape, "\n-", Y_train.shape)
    print("Validation\n-", X_val.shape, "\n-", Y_val.shape)
    print("Testing\n-", X_test.shape, "\n-", Y_test.shape)

    train_data_gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, 
                                        height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
                                        horizontal_flip=True,
                                        preprocessing_function=preprocessing_function)
    train_data = train_data_gen.flow(
        x=X_train, y=Y_train,
        batch_size=batch_size,
        shuffle=True)

    val_data_gen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    val_data = val_data_gen.flow(
        x=X_val, y=Y_val,
        batch_size=batch_size,
        shuffle=True)

    test_data_gen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    test_data = test_data_gen.flow(
        x=X_test, y=Y_test,
        batch_size=batch_size,
        shuffle=True)

    return train_data, val_data, test_data


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


""" Load RAF Dataset """
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


""" Load biased RAF Dataset """
def load_biased_RAF(folderpath, image_dir_name = 'images/', target_img_size=(100, 100, 3)):
    """
    @params		folderpath, image_dir_name, target_img_size
    @returns	train_data, test_data

    folderpath: path where the data is stored with the following structure
    image_dir_name: name of directory of the biased data (e.g. 'male/')
    /labels.txt
    /image_dir_name
      /test_0001.jpg
      /test_0002.jpg
      /test_0003.jpg
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
        filename = name + '.' + t
        exists = Path(folderpath + image_dir_name + filename)
        if not exists.is_file(): continue

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

    train_data = train_data_gen.flow_from_dataframe(train_set, directory = folderpath + image_dir_name, batch_size=32, 
                                                    color_mode='rgb', shuffle=True, class_mode='categorical',
                                                    target_size=target_img_size[:2])

    test_data = test_data_gen.flow_from_dataframe(test_set, directory = folderpath + image_dir_name, batch_size=32, 
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


if __name__ == "__main__":
    RAF_DIR = './data/RAF/'
    load_RAF_targetvector(RAF_DIR)