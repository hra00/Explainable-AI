import math
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import tight_layout

labels_raf = {  0:  'Anger',
                1:  'Disgust',
                2:  'Fear',
                3:  'Happiness',
                4:  'Neutral',
                5:  'Sadness',
                6:  'Surprise'
            }
labels_fer = {  0:  'NF',
                1:  'anger',
                2:  'comtempt',
                3:  'disgust',
                4:  'fear',
                5:  'happiness',
                6:  'neutral',
                7:  'sadness',
                8:  'surprise',
                9:  'unknown'
            }
raw_labels_raf = {  1:  'Surprise',
                    2:  'Fear',
                    3:  'Disgust',
                    4:  'Happiness',
                    5:  'Sadness',
                    6:  'Anger',
                    7:  'Neutral'
                }
raw_labels_fer = {  1: 'neutral',
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

def img_to_tensor(img_path, img_size=(100,100,3), RGB=True):
    img = cv2.imread(img_path, 1)[...,::-1]
    img_tensor = cv2.resize(img, img_size[:2])
    img_tensor = np.float32(img_tensor) / 255 if RGB else img_tensor
    return np.float32(img_tensor)

def img_to_input_tensor(img_paths, img_size=(100,100,3), RGB=True):
    return np.array([img_to_tensor(img_path, img_size, RGB).astype(np.float32) for img_path in img_paths])

def get_img_list_raf(emotion : str, num_imgs=None):
    img_list = []
    folderpath = './data/RAF/'
    imgpath = './data/RAF/images/'
    labelling_list = open(folderpath + 'labels.txt', 'r').read().strip().split('\n')
    for name_label in labelling_list:
        image_name, label = name_label.split(' ')
        if raw_labels_raf[int(label)] == emotion and 'test' == image_name[:4]:
            name, fileend = image_name.split('.')
            filename = imgpath + name + '_aligned.' + fileend
            img_list.append(filename)
    if num_imgs:
        img_list = random.sample(img_list,num_imgs)
    return img_list

def get_img_list_fer(emotion : str, num_imgs=None):
    img_list = []
    csv_path = './data/ferplus2013/fer2013new.csv'
    test_dir = './data/ferplus2013/images/FER2013Test/'
    labelling_list = open(csv_path, 'r').read().strip().split('\n')
    for name_label in labelling_list:
        usage, file_name, neutral, happiness, surprise, sadness, anger, disgust, fear, contempt, unknown, NF = name_label.split(',')
        category = [neutral, happiness, surprise, sadness, anger, disgust, fear, contempt, unknown, NF]
        label = raw_labels_fer[np.argmax(category)+1]
        if label == emotion and usage=='PrivateTest':
            filepath = test_dir + file_name
            img_list.append(filepath)
    if num_imgs:
        img_list = random.sample(img_list, num_imgs)
    return img_list

def scale_cam(cam, img_shape):
    cam = np.clip(cam, 0, np.max(cam)) / np.max(cam)
    cam = cv2.resize(cam, img_shape[:2])
    return np.float32(cam)

# TODO : labels, blended heatmap
def pp_images(img_tensors, heatmaps=None, preds=None, labels=None, RGB = None, alpha = 0.5):
    img_tensors = [img_tensor for img_tensor in img_tensors]
    num_imgs = len(img_tensors)
    W = min(num_imgs, 10)
    H = math.ceil(num_imgs / W)
    _, axs = plt.subplots(H, W, figsize=(60,W*3))
    for i in range(num_imgs):
        img_tensors[i] = img_tensors[i]/255 if RGB else img_tensors[i]
        if H == 1:
            axs[i % W].imshow(img_tensors[i])
            if heatmaps:
                # axs[i//W][i%W].set_title(labels[preds[i]])
                axs[i % W].imshow(heatmaps[i], cmap='jet', alpha=alpha)
        else:
            axs[i//W][i%W].imshow(img_tensors[i])
            if heatmaps:
                #axs[i//W][i%W].set_title(labels[preds[i]])
                axs[i//W][i%W].imshow(heatmaps[i], cmap='jet', alpha=alpha)
    tight_layout()
    plt.show()


def pp_blended_heatmaps(heatmaps:[[np.array]]):
    blended = [scale_cam(sum(heatmap), heatmaps[0][0].shape) for heatmap in heatmaps]
    pp_images(blended)
