import dlib
import cv2
import numpy as np
from renderFace import renderFace

PREDICTOR_PATH="../../common/resources/shape_predictor_68_face_landmarks.dat"

faceDetector=dlib.get_frontal_face_detector()
landmarkDetector=dlib.shape_predictor(PREDICTOR_PATH)

im=cv2.imread("../data/images/family.jpg")

faceRects=faceDetector(im,0)
print("面部数量为:",len(faceRects))

landmarksAll=[]

for i in range(0, len(faceRects)):
    newRect=dlib.rectangle(int(faceRects[i].left()), int(faceRects[i].top()),int(faceRects[i].right()), int(faceRects[i].bottom()))
    landmarks= landmarkDetector(im, newRect)
    landmarksAll.append(landmarks)
    renderFace(im, landmarks)


cv2.imshow("Detection", im)
cv2.waitKey(0)
cv2.destroyAllWindows()