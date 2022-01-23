import cv2
import numpy as np
import pathlib
from matplotlib import pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import tensorflow as tf
import tensorflow.keras.backend as K

from IPython.display import clear_output, Image, display, HTML


class VideoDemonstrator():
    def __init__(self, visualizer, VIDEO_PATH=0):
        self.vis = visualizer
        self.VIDEO_PATH = VIDEO_PATH
        self.PADDING = 0 # CAUTION: padding can lead to errors, TODO: fix
        self.INTENSITY = 0.3
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        # load face detection dnn
        modelFile = str(pathlib.Path().absolute()) + "/models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
        configFile = str(pathlib.Path().absolute()) + "/models/face_detection_model/deploy.prototxt.txt"
        self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    

    def start_demonstrator(self, visualizer=None):
        if visualizer is not None: self.vis = visualizer

        cap = cv2.VideoCapture(self.VIDEO_PATH)

        while(True):
            ret, img = cap.read()
            if ret == True:
                img = cv2.resize(img, None, fx=0.5, fy=0.5)
                height, width = img.shape[:2]
                img2 = img.copy()
                class_name = 'no_class'

                # detect faces in the image
                blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                                1.0, (300, 300), (104.0, 117.0, 123.0))
                self.net.setInput(blob)
                faces = self.net.forward()
                
                # display faces on the original image
                for i in range(faces.shape[2]):
                    confidence = faces[0, 0, i, 2]
                    if confidence > 0.5:
                        box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
                        (x, y, x1, y1) = box.astype("int")
                        roi_face = img2[y-self.PADDING:y1+self.PADDING, x-self.PADDING:x1+self.PADDING]
                        cv2.rectangle(img2, (x, y), (x1, y1), (0, 0, 255), 2)
                        
                        #cv2.imshow("prep_image", prep_img)
                        class_name, heatmap = self.vis.visualize(roi_face, (x1-x, y1-y))
                        # TODO: add try block with padding...
                        img2[y:y1, x:x1, :] = img2[y:y1, x:x1, :] - heatmap*self.INTENSITY
                
                cv2.putText(img2, class_name, (30, 30), self.FONT, 1, (255, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow("image with heatmap", img2)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()



class VisualizerInterface:
    def __init__(self, MODEL_PATH, IMG_SIZE, class_labels):
        self.model = load_model(MODEL_PATH)
        self.IMG_SIZE = IMG_SIZE
        self.class_labels = class_labels

    def preprocess(self, image):
        """ Return preprocessed image """
        pass

    def visualize(self, image, OUTPUT_SIZE):
        """ Return the predicted class and the map or other overlay for the image """
        pass


class VisualizerGradCAM(VisualizerInterface):
    def __init__(self, MODEL_PATH, IMG_SIZE, class_labels, last_conv_size=(5, 5), debug=False):
        super().__init__(MODEL_PATH, IMG_SIZE, class_labels)
        if debug: self.model.summary()
        self.last_conv_size = last_conv_size

    def preprocess(self, img):
        img = cv2.resize(img, self.IMG_SIZE[:2])
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor = preprocess_input(img_tensor)
        return img_tensor

    def visualize(self, img, OUTPUT_SIZE):
        image_preprocessed = self.preprocess(img)

        with tf.GradientTape() as tape:
            last_conv_layer = self.model.get_layer('conv2d_7')
            iterate = tf.keras.models.Model([self.model.inputs], [self.model.output, last_conv_layer.output])
            model_out, last_conv_layer = iterate(image_preprocessed)
            class_out = model_out[:, np.argmax(model_out[0])]
            grads = tape.gradient(class_out, last_conv_layer)
            pooled_grads = K.mean(grads, axis=(0, 1, 2))
        
        model_out = self.model.predict(image_preprocessed)
        class_name = self.class_labels[np.argmax(model_out)][0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap.reshape(self.last_conv_size)
        heatmap = cv2.resize(heatmap, OUTPUT_SIZE)
        heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        return class_name, heatmap


# From here on from Johannes 
class VisualizerMasking(VisualizerInterface):
    def __init__(self, MODEL_PATH, IMG_SIZE, class_labels, WINDOW_SIZE, debug=True):
        super().__init__(MODEL_PATH, IMG_SIZE, class_labels)
        if debug: self.model.summary()
        self.num_classes  = len(self.class_labels)
        self.WINDOW_SIZE = WINDOW_SIZE
        self.STEP_SIZE = self.WINDOW_SIZE[0]
    
    def preprocess(self, image, resize_size):
        image = cv2.resize(image, resize_size)
        image = (image.astype("float32") / 255)
        return image

    def get_prediction(self, image_batch):
        preds = self.model.predict(image_batch)
        #preds = np.random.randn(image_batch[0], 7)
        return preds

    def sliding_windows_(self, image):
        h, w = image.shape[:2]
        window_h, window_w = self.WINDOW_SIZE
        
        for y in range(0, h, self.STEP_SIZE):
            for x in range(0, w, self.STEP_SIZE):
                patch = image[y : y + window_h, x : x + window_w]
                if patch.shape[0] == window_h and patch.shape[1] == window_w:
                    yield (x, y, patch)
                    
    """def softmax_2D(matrix):
        denominator = np.sum(matrix.flatten())
        softmax_matrix = np.zeros(matrix.shape)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                softmax_matrix[row, col] = matrix[row, col] / denominator
        return softmax_matrix"""

    def visualize(self, image, OUTPUT_SIZE):
        # preprocess image
        image_preprocessed = self.preprocess(image, self.IMG_SIZE[:2])
        
        # predict single image without masking
        base_pred = self.model.predict(np.expand_dims(image_preprocessed, axis=0))
        label = emotion_labels[np.argmax(base_pred)+1]
        #print(f"Base prediction: {round(np.max(base_pred)*100, 2)}% at position {np.argmax(base_pred)} ({emotion_labels[np.argmax(base_pred)+1]})")
        
        # create masked images
        masked_imgs = []
        for x, y, patch in self.sliding_windows_(image_preprocessed):
            clone = image_preprocessed.copy()
            cv2.rectangle(clone, (x, y), (x + self.WINDOW_SIZE[0], y + self.WINDOW_SIZE[0]), (0, 0, 0), -1)
            masked_imgs.append(clone)
        masked_imgs = np.array(masked_imgs)
        #print("Masked images:\t", masked_imgs.shape)
        
        # get predictions for the masked images
        preds = self.model.predict(masked_imgs)
        #print("Predictions:\t", preds.shape)
        
        # create feature map based on predictions
        feature_maps = np.zeros( ((self.num_classes,) + image_preprocessed.shape[:2]) )
        for i, (x, y, patch) in enumerate(self.sliding_windows_(image_preprocessed)):
            # for each emotion make feature map
            activation_array = base_pred - preds[i]
            for emotion_idx in range(self.num_classes):
                #activation = 1-preds[i, emotion_idx]
                activation = activation_array[0,emotion_idx]
                feature_maps[emotion_idx, y:y + self.WINDOW_SIZE[0], x:x + self.WINDOW_SIZE[0]] = activation
        #print("Feature maps:\t", feature_maps.shape)
        
        emotion_heatmap = feature_maps[np.argmax(base_pred)]
        emotion_heatmap = cv2.resize(emotion_heatmap, OUTPUT_SIZE)
        heatmap_img = cv2.applyColorMap(np.uint8(255*emotion_heatmap), cv2.COLORMAP_JET)
        
        return label, heatmap_img


if __name__ == "__main__":
    """ GradCAM visualization """
    """
    model_config = ('./models/pretrained_emotion/Hyeri Raf_model.h5', (100, 100, 3), \
        [('Anger', 0), ('Disgust', 1), ('Fear', 2), ('Happiness', 3), ('Neutral', 4), ('Sadness', 5), ('Surprise', 6)], (11, 11))
        # ('NF', 0), ('anger', 1), ('comtempt', 2), ('disgust', 3), ('fear', 4), ('happiness', 5), ('neutral', 6), ('sadness', 7), ('surprise', 8), ('unknown', 9)

    vis = VisualizerGradCAM(model_config[0], model_config[1], model_config[2], model_config[3])
    #"""
    

    """ Masking visualization"""
    #"""
    emotion_labels = {  1: 'Anger',
                        2: 'Disgust',
                        3: 'Fear',
                        4: 'Happiness',
                        5: 'Neutral',
                        6: 'Sadness',
                        7: 'Surprise'
                    }
    model_config = ('./models/pretrained_emotion/Hyeri Raf_model.h5', (100, 100, 3), emotion_labels, (10, 10))
    vis = VisualizerMasking(model_config[0], model_config[1], model_config[2], model_config[3])
    #"""

    demo = VideoDemonstrator(vis)
    demo.start_demonstrator()