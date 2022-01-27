import numpy as np
from utils.image import scale_cam

def get_eigencam(activation):
    reshaped_activation = activation.reshape(activation.shape[3], -1).transpose()
    reshaped_activation -= reshaped_activation.mean(axis=0)
    _, _, VT = np.linalg.svd(reshaped_activation, full_matrices=True)
    projection = [activation[:, :, :, i] * VT[0, i] for i in range(activation.shape[3])]
    cam = np.zeros(shape=activation.shape[1:3])
    for filters in projection:
        cam += filters[0]
    return np.float32(cam)


def get_EigenCAM(vis, img_tensors: np.ndarray) -> [[np.ndarray], [int]]:
    activations = vis.activation_model.predict(img_tensors)
    cams = [get_eigencam(activation[np.newaxis, ...]) for activation in activations]
    predicts = [np.argmax(pred) for pred in vis.model.predict(img_tensors)]
    heatmaps = [scale_cam(cam, vis.img_shape) for cam in cams]
    return heatmaps, predicts