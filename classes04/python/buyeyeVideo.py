import cv2
import dlib
import time
import argparse
import numpy as np

FACE_DOWNSAMPLE_RATION = 1.5
RESIZE_HEIGHT = 360

modelPath = "../../common/resources/zxm_shape_predictor_70_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

bulgeAmount = 2
radius = 10
print("注意：bugeyeVideo.py bulgeAmount:{} radius:{}".format(bulgeAmount, radius))

def barrel(src, k):
    w = src.shape[1]
    h = src.shape[0]

    Xu, Yu = np.meshgrid(np.arange(w), np.arange(h))

    Xu = np.float32(Xu)/w - 0.5
    Yu = np.float32(Yu)/h - 0.5

    XuSquare = np.square(Xu)
    YuSquare = np.square(Yu)

    r = np.sqrt(XuSquare + YuSquare)

    rn = np.minimum(r, r+np.multiply((np.power(r,k)-r), np.cos(np.pi * r) ))

    Xd = w *(cv2.divide(np.multiply(rn, Xu), r) + 0.5)
    Yd = h *(cv2.divide(np.multiply(rn, Yu), r) + 0.5)

    dst = cv2.remap(src, Xd, Yd, cv2.INTER_CUBIC)
    return dst

def getLandmarks(im):
    imSmall = cv2.resize(im, None, fx=1.0/FACE_DOWNSAMPLE_RATION, fy=1.0/FACE_DOWNSAMPLE_RATION, interpolation=cv2.INTER_LINEAR)
    rects = detector(imSmall, 0)
    if len(rects) == 0:
        return 1
    
    newRect = dlib.rectangle( int(rects[0].left()*FACE_DOWNSAMPLE_RATION),
                              int(rects[0].top()*FACE_DOWNSAMPLE_RATION),
                              int(rects[0].right()*FACE_DOWNSAMPLE_RATION),
                              int(rects[0].bottom()*FACE_DOWNSAMPLE_RATION))
    points = []
    [points.append((p.x, p.y)) for p in predictor(im, newRect).parts()]
    return points


ap = argparse.ArgumentParser()
ap.add_argument("-r", "--radius", help="眼睛的半径默认为:30")
ap.add_argument("-b", "--bulge", help="bugle的值默认为:2")

args = vars(ap.parse_args())

if(args["radius"]):
    radius=args["radius"]
if(args["bulge"]):
    bulgeAmount = float(args["bulge"])

capture = cv2.VideoCapture(0)
while(1):
    try:
        ret, src = capture.read()
        height, width = src.shape[:2]
        IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
        src = cv2.resize(src, None, fx=1.0/IMAGE_RESIZE, fy=1.0/IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)
        landmarks = getLandmarks(src)
        if landmarks == 1:
            cv2.putText(src, "Unable to detect face", (10,50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,0,255), 1.0, cv2.LINE_AA)
            cv2.imshow("Bug Eye Demo", src)
            if(cv2.waitKey(1)&0xFF == 27):
                break
            continue

        roiEyeLeft = [landmarks[37][0] - radius, landmarks[37][1]-radius,
                      (landmarks[40][0] - landmarks[37][0] + 2*radius),
                      (landmarks[41][1] - landmarks[37][1] + 2*radius)]
        
        roiEyeRight = [landmarks[43][0] - radius, landmarks[43][1]-radius,
                      (landmarks[46][0] - landmarks[43][0] + 2*radius),
                      (landmarks[47][1] - landmarks[43][1] + 2*radius)]
        
        output = np.copy(src)

        eyeRegion = src[roiEyeLeft[1]:roiEyeLeft[1]+roiEyeLeft[3], roiEyeLeft[0]:roiEyeLeft[0]+roiEyeLeft[2]]
        eyeRegion = barrel(eyeRegion, bulgeAmount)
        output[roiEyeLeft[1]:roiEyeLeft[1]+roiEyeLeft[3], roiEyeLeft[0]:roiEyeLeft[0]+roiEyeLeft[2]]=eyeRegion

        eyeRegion = src[roiEyeRight[1]:roiEyeRight[1]+roiEyeRight[3], roiEyeRight[0]:roiEyeRight[0]+roiEyeRight[2]]
        eyeRegion = barrel(eyeRegion, bulgeAmount)
        output[roiEyeRight[1]:roiEyeRight[1]+roiEyeRight[3], roiEyeRight[0]:roiEyeRight[0]+roiEyeRight[2]]=eyeRegion

        cv2.imshow("Bug Eye Demo", output)

        if(cv2.waitKey(1)&0xFF)==27:
            break
        
    except Exception as e:
        print(e)

capture.release()
cv2.destroyAllWindows()
