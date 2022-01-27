import numpy as np
from cams.visualizer import Visualizer
from utils.image import scale_cam

# https://arxiv.org/abs/2008.00299

class EigenCAM(Visualizer):
    def __init__(self, model_name, last_conv_name, img_shape):
        super(EigenCAM, self).__init__(model_name, last_conv_name, img_shape)

    def get_eigencam(self, activation):
        reshaped_activation = activation.reshape(activation.shape[3], -1).transpose()
        reshaped_activation -= reshaped_activation.mean(axis=0)
        _, _, VT = np.linalg.svd(reshaped_activation, full_matrices=True)
        projection = [activation[:, :, :, i] * VT[0, i] for i in range(activation.shape[3])]
        cam = np.zeros(shape=activation.shape[1:3])
        for filters in projection:
            cam += filters[0]
        return np.float32(cam)

    def get_CAM(self, img_tensors: np.ndarray) -> [[np.ndarray], [int]]:
        activations = self.activation_model.predict(img_tensors)
        cams = [self.get_eigencam(activation[np.newaxis, ...]) for activation in activations]
        predicts = [np.argmax(pred) for pred in self.model.predict(img_tensors)]
        heatmaps = [scale_cam(cam, self.img_shape) for cam in cams]
        return heatmaps, predicts


