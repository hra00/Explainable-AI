import numpy as np
import tensorflow as tf
from cams.visualizer import Visualizer
from utils.image import scale_cam

# TODO : Batch Implementation..?
class GradCAM(Visualizer):
    def __init__(self, model_name, last_conv_name, img_shape):
        super(GradCAM, self).__init__(model_name, last_conv_name, img_shape)

    def get_CAM(self, img_tensors: np.ndarray) -> [[np.ndarray], [int]]:
        heatmaps = []
        preds = []
        for img in img_tensors:
            img = img[np.newaxis, ...]
            heatmap, pred = self.get_gradcam(img)
            heatmaps.append(heatmap)
            preds.append(pred)
        return heatmaps, preds

    def get_gradcam(self, input_tensor):
        pred = np.argmax(self.model.predict(input_tensor)[0])
        with tf.GradientTape() as tape:
            inputs = input_tensor
            last_conv_layer_output = self.activation_model(inputs)
            tape.watch(last_conv_layer_output)
            preds = self.classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        gradcam = np.mean(last_conv_layer_output, axis=-1)
        return scale_cam(gradcam, self.img_shape), pred
