import numpy as np
import tensorflow as tf
from utils.image import scale_cam

def get_gradcam(vis, input_tensor):
    pred = np.argmax(vis.model.predict(input_tensor)[0])
    with tf.GradientTape() as tape:
        inputs = input_tensor
        last_conv_layer_output = vis.activation_model(inputs)
        tape.watch(last_conv_layer_output)
        preds = vis.classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    gradcam = np.mean(last_conv_layer_output, axis=-1)
    return scale_cam(gradcam, vis.img_shape), pred

def get_GradCAM(vis, img_tensors: np.ndarray) -> [[np.ndarray], [int]]:
    heatmaps = []
    preds = []
    for img in img_tensors:
        img = img[np.newaxis, ...]
        heatmap, pred = get_gradcam(vis, img)
        heatmaps.append(heatmap)
        preds.append(pred)
    return heatmaps, preds

