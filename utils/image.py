import math
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import tight_layout
from retinaface import RetinaFace as rf

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

def img_to_tensor(img_path, img_size=(100,100,3)):
    img = cv2.imread(img_path, 1)[...,::-1]
    img_tensor = cv2.resize(img, img_size[:2])
    return np.float32(img_tensor)

def img_to_input_tensor(img_paths, img_size=(100,100,3)):
    img_tensors = [img_to_tensor(img_path, img_size).astype(np.float32) for img_path in img_paths]
    img_tensors_rgb = np.array([img / 255 for img in img_tensors])
    img_tensors_not_rgb = np.array(img_tensors)
    return np.array(img_tensors)/255, img_tensors_rgb, img_tensors_not_rgb

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

def aligned_tensor(img_paths, img_size=(100,100,3)):
    img_tensors = [cv2.resize(rf.extract_faces(img_path=img, align=True)[0], img_size[:2]) for img in img_paths]
    img_tensors_rgb = np.array([img / 255 for img in img_tensors])
    img_tensors_not_rgb = np.array(img_tensors)
    return img_tensors, img_tensors_rgb, img_tensors_not_rgb


def pp_images(img_tensors, heatmaps=None, preds=None, labels=None, RGB = None, alpha = 0.5, figsize=(50,10), wd=10, fontsize=37, axis=True):
    img_tensors = [img_tensor for img_tensor in img_tensors]
    num_imgs = len(img_tensors)
    W = min(num_imgs, wd)
    H = math.ceil(num_imgs / W)
    _, axs = plt.subplots(H, W, figsize=figsize)
    for i in range(num_imgs):
        img_tensors[i] = img_tensors[i]/255 if RGB else img_tensors[i]
        ax = axs[i % W] if H==1 else axs[i//W][i%W]
        ax.imshow(img_tensors[i])
        if labels and preds:
            ax.set_title('{label} ({percent}%)'.format(label=labels[preds[i][0]], percent=preds[i][1]), fontsize=fontsize)
        elif labels and not preds:
            ax.set_title(labels[i])
        if heatmaps:
            ax.imshow(heatmaps[i], cmap='jet', alpha=alpha)
        if not axis:
            ax.axis('off')
    tight_layout()
    plt.show()

def pp_blended_heatmaps(heatmaps:[[np.array]], figsize=None):
    blended = [scale_cam(sum(heatmap), heatmaps[0][0].shape) for heatmap in heatmaps]
    pp_images(blended, figsize=figsize)


def save_results(save_path, imgs, heatmaps, titles):
    for img, title in zip(imgs, titles[:6]):
        plt.imshow(img)
        plt.savefig(save_path + title + '.png', dpi=300, bbox_inches='tight')
    for img, heatmap, title in zip(imgs * 5, heatmaps, titles[6:]):
        plt.imshow(img)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.savefig(save_path + title + '.png', dpi=300, bbox_inches='tight')