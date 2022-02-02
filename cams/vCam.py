import numpy as np
from utils.image import scale_cam

def get_CAM(vis, img_tensors: np.ndarray, preprocess, start_idx):
    image_batch = np.array([vis.preprocess(img) for img in img_tensors]) if preprocess else img_tensors

    # predict
    model_out, feature_maps = vis.cam_model.predict(image_batch)

    # make CAMs
    weights = vis.model.layers[-1].weights[0]
    # for each image repeat
    CAMs = []
    for idx in range(image_batch.shape[0]):
        max_idx = np.argmax(np.squeeze(model_out[idx])) if not start_idx else start_idx
        # print(f"Maximum index {max_idx} with confidence {model_out[idx][max_idx] * 100:.2f}%")

        winning_weights = weights[:, max_idx]
        CAM = np.sum(feature_maps[idx] * winning_weights, axis=2)

        CAMs.append(CAM)
    heatmaps = [scale_cam(np.float32(cam), vis.img_shape) for cam in CAMs]
    """
    for idx, img in enumerate(images):
        #heatmap = cv2.resize(CAM, self.img_shape[:2])
        heatmaps.append(scale_cam(CAM, self.img_shape))
    """
    if start_idx:
        preds = [(start_idx, '%0.2f' % (pred[start_idx] * 100)) for pred in model_out.tolist()]
    else:
        preds = [(pred.index(max(pred)), '%0.2f' % (max(pred) * 100)) for pred in model_out.tolist()]
    return heatmaps, preds
