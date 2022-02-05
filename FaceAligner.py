import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib

# load face detector
face_detector = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

# load landmark detector
landmark_model = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

# eye facial landmarks
FACIAL_LANDMARKS_IDXS = {
    "left_eye": [36, 37, 38, 39, 40, 41],
    "right_eye": [42, 43, 44, 45, 46, 47]
}

# convert landmarks shape to numpy array
def shape2numpy(shape):
    xy = [(shape.part(i).x, shape.part(i).y,) for i in range(68)]
    return np.array(xy, dtype='float32')


def align_image(image, leftEyePos=(0.32, 0.32), faceSize=200, minSizeFaceDetector=(100, 100)):
    # convert to grayscale for face detector
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces
    bboxes = face_detector.detectMultiScale(gray, 1.1, 2, minSize=minSizeFaceDetector)

    # if no face was found return empty array
    if len(bboxes) == 0: return []
    
    # take face if there is only one
    if len(bboxes) == 1: bbox_idx = 0
    # take largest face otherwise
    else: bbox_idx = np.argmax(bboxes[:, 2])

    # get bounding box coordinates and width/height of face
    (x, y, w, h) = bboxes[bbox_idx]

    # get face crop
    face = image[y:y + h, x:x + w]

    # get landmarks
    shape = landmark_model(face, dlib.rectangle(0, 0, face.shape[0], face.shape[1]))
    shape_np = shape2numpy(shape)

    # only get eyes
    leftEyePts = shape_np[FACIAL_LANDMARKS_IDXS["left_eye"]]
    rightEyePts = shape_np[FACIAL_LANDMARKS_IDXS["right_eye"]]

    # get center points
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    # compute angle
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # get desired eye position
    desiredRightEyeX = 1.0 - leftEyePos[0]

    # compute scaling factor
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - leftEyePos[0])
    desiredDist *= faceSize
    scale = desiredDist / dist

    # compute center between both eyes
    eyesCenter = (
        (leftEyeCenter[0] + rightEyeCenter[0]) // 2,
        (leftEyeCenter[1] + rightEyeCenter[1]) // 2
    )

    # get rotation matrix
    M = cv2.getRotationMatrix2D((int(eyesCenter[0]), int(eyesCenter[1])), angle, scale)

    # update the translation component of the matrix
    tX = faceSize * 0.5
    tY = faceSize * leftEyePos[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply affine transformation
    (w, h) = (faceSize, faceSize)
    output = cv2.warpAffine(face, M, (w, h), flags=cv2.INTER_CUBIC)

    return output



if __name__ == "__main__":

    # load image
    img = cv2.imread('./data/multiple.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # align it
    aligned_img = align_image(img)

    # show results
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(aligned_img)
    plt.title('Aligned')
    plt.axis('off')
    plt.show()