emotion_labels_RAF_impr = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
emotion_labels_FERplus_impr = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
emotion_labels_raf_h = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
emotion_labels_ferplus_h = ['NF', 'anger', 'comtempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'unknown']

def get_labels(model_name):
    model_path = './models/'
    if model_name == 'FERplus-impr-std_0124-1040_weights.h5' or 'FERplus-biased_0208-1922_weights.h5':
        return emotion_labels_FERplus_impr
    else:
        return emotion_labels_RAF_impr