import numpy as np
import tensorflow as tf
from utils.image import scale_cam

def get_gradcam(vis, input_tensor, start_idx):
    model_out = vis.model.predict(input_tensor)[0]
    with tf.GradientTape() as tape:
        inputs = input_tensor
        last_conv_layer_output = vis.activation_model(inputs)
        tape.watch(last_conv_layer_output)
        preds = vis.classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0]) if not start_idx else start_idx
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    gradcam = np.mean(last_conv_layer_output, axis=-1)
    return scale_cam(gradcam, vis.img_shape), model_out

def get_GradCAM(vis, img_tensors: np.ndarray, preprocess, start_idx) -> [[np.ndarray], [int]]:
    if preprocess:
        img_tensors = np.array([vis.preprocess(img) for img in img_tensors])
    heatmaps = []
    preds = []
    for img in img_tensors:
        img = img[np.newaxis, ...]
        heatmap, model_out = get_gradcam(vis, img, start_idx)
        heatmaps.append(heatmap)
        preds.append(model_out)
    return heatmaps, [(np.argmax(pred), '%0.2f' % (max(pred) * 100)) for pred in preds]

