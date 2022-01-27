import numpy as np
from utils.image import scale_cam

from cams.visualizer import Visualizer

class CAM(Visualizer):
    def __init__(self, model_name, last_conv_name, img_shape):
        super(CAM, self).__init__(model_name, last_conv_name, img_shape, self.preprocess)

    def get_CAM(self, images):
        image_batch = np.array([self.preprocess(img) for img in images])

        # predict
        model_out, feature_maps = self.cam_model.predict(image_batch)

        # make CAMs
        weights = self.model.layers[-1].weights[0]
        # for each image repeat
        CAMs = []
        for idx in range(image_batch.shape[0]):
            max_idx = np.argmax(np.squeeze(model_out[idx]))
            #print(f"Maximum index {max_idx} with confidence {model_out[idx][max_idx] * 100:.2f}%")

            winning_weights = weights[:, max_idx]
            CAM = np.sum(feature_maps[idx] * winning_weights, axis=2)

            CAMs.append(CAM)
        heatmaps = [scale_cam(np.float32(cam), self.img_shape) for cam in CAMs]
        """
        for idx, img in enumerate(images):
            #heatmap = cv2.resize(CAM, self.img_shape[:2])
            heatmaps.append(scale_cam(CAM, self.img_shape))
        """
        return heatmaps, [(pred.index(max(pred)), '%0.2f' % (max(pred)*100)) for pred in model_out.tolist()]

    def preprocess(self, x):
        mean = [129.4432, 129.4432, 129.4432]
        std = [64.87448751, 64.87448751, 64.87448751]
        # ensure image format
        x = np.array(x, dtype='float32')
        # normalize
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
        return x