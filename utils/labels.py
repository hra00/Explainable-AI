emotion_labels_RAF_impr = ['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']
emotion_labels_FERplus_impr = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
emotion_labels_raf_h = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
emotion_labels_ferplus_h = ['NF', 'anger', 'comtempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'unknown']

def get_labels(model_name):
    model_path = './models/'
    if model_name == 'FERplus-impr-std_0124-1040_weights.h5':
        return emotion_labels_FERplus_impr
    elif model_name == 'RAF-impr-std_0124-1008_weights.h5':
        return emotion_labels_RAF_impr
    elif model_name == 'RAF-bias-female_0203-1408_weights.h5':
        return emotion_labels_RAF_impr
    elif model_name == 'RAF-bias-male_0203-1144_weights.h5':
        return emotion_labels_RAF_impr
    elif model_name == 'model_raf_h.h5':
        return emotion_labels_raf_h
    elif model_name == 'model_ferplus_h.h5':
        return emotion_labels_raf_h