import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.ModelGenerator import get_base_model2
from cams.vCam import get_CAM
from cams.eigenCam import get_EigenCAM
from cams.gradCam import get_GradCAM

class Visualizer:
    def __init__(self,
                 model_name,
                 last_conv_name,
                 img_shape,
                 preprocess=None):
        self.model_name = model_name
        self.img_shape = img_shape
        self.model = self.loadModel()
        self.last_conv_name = last_conv_name
        self.last_conv_layer = self.model.get_layer(self.last_conv_name)
        self.cam_model = self.getCamModel()
        self.activation_model = self.getActivationModel()
        self.preprocess = preprocess
        self.classifier_model = self.getClassifierModel()

    def loadModel(self):
        if self.model_name == 'FERplus-impr-std_0124-1040_weights.h5':
            model = get_base_model2(self.img_shape)
            model.add(tf.keras.layers.Dense(7, activation='softmax', name="softmax"))
            model.load_weights('./models/' + self.model_name)
        else:
            model = load_model('./models/' + self.model_name)
        return model

    def getCamModel(self):
        cam_model = tf.keras.Model( inputs=self.model.inputs,
                                    outputs=[self.model.output, self.last_conv_layer.output])
        return cam_model

    def getActivationModel(self):
        activation_model =tf.keras.Model(self.model.inputs, self.last_conv_layer.output)
        return activation_model

    def getClassifierModel(self):
        classifier_input = tf.keras.Input(shape=self.activation_model.output.shape[1:])
        x = classifier_input
        layer_names = [layer.name for layer in self.model.layers]
        last_conv_idx = layer_names.index(self.last_conv_name)
        for layer_name in layer_names[last_conv_idx:]:
            x = self.model.get_layer(layer_name)(x)
        classifier_model = tf.keras.Model(classifier_input, x)
        return classifier_model

    def getCAM(self, input_tensor):
        return get_CAM(self, input_tensor)

    def getEigenCAM(self, input_tensor):
        return get_EigenCAM(self, input_tensor)

    def getGradCAM(self, input_tensor):
        return get_GradCAM(self, input_tensor)




