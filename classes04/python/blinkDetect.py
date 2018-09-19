import cv2
import dlib
import time
import sys
import numpy as np

FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 360

thresh = 0.43

# 全局变量
modelPath = "../../common/resources/zxm_shape_predictor_70_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

# 眼睛的dlib索引点
leftEyeIndex = [36, 37, 38, 39, 40, 41]
rightEyeIndex = [42, 43, 44, 45, 46, 47]

# 计算FPS的变量
blinkCount = 0
drowsy = 0
state =0
blinkTime = 0.2  #200ms
drowsyTime = 1.0 #1000ms

# 检查眼睛的状态
def checkEyeStatus(landmarks):
    """
    检查眼睛是睁开还是闭上
    参数:
    =============
    landmarks:形状的关键特征点
    返回:
    =============
    1->眼镜睁开；0->眼睛闭上;
    """
    pass

    # 创建一个黑色的遮罩用来截取眼镜区域
    mask = np.zeros(frame.shape[:2], dtype=np.float32)

    hullLeftEye=[]
    for i in range(0, len(leftEyeIndex)):
        hullLeftEye.append((landmarks[leftEyeIndex[i]][0], landmarks[leftEyeIndex[i]][1]))
    cv2.fillConvexPoly(mask, np.int32(hullLeftEye), 255)

    hullRightEye = []
    for i in range(0, len(rightEyeIndex)):
        hullRightEye.append((landmarks[rightEyeIndex[i]][0], landmarks[rightEyeIndex[i]][1]))
    cv2.fillConvexPoly(mask, np.int32(hullRightEye), 255)

    lenLeftEyeX = landmarks[leftEyeIndex[3]][0] - landmarks[leftEyeIndex[0]][0]
    lenLeftEyeY = landmarks[leftEyeIndex[3]][1] - landmarks[leftEyeIndex[0]][1]

    lenLeftEyeSquare = lenLeftEyeX*lenLeftEyeX + lenLeftEyeY*lenLeftEyeY

    eyeRegionCount = cv2.countNonZero(mask)

    normalizedCount = eyeRegionCount/np.float32(lenLeftEyeSquare)

    eyeStatus = 1
    if(normalizedCount < thresh):
        eyeStatus = 0

    return eyeStatus

def checkBlinkStatus(eyeStatus):
    global state, blinkCount, drowsy

    # 睁开状态和假眨眼状态
    if(state >=0 and state <= falseBlinkLimit):
        # 如果眼睛是睁开的
        if(eyeStatus):
            state=0
        # 如果还是闭着的
        else:
            state +=1
    elif(state>falseBlinkLimit and state <= drowsyLimit):
        if(eyeStatus):
            state = 0
            blinkCount += 1
        else:
            state +=1
    else:
        if(eyeStatus):
            state = 0
            blinkCount +=1
            drowsy =0
        else:
            drowsy =1

def getLandmarks(im):
    imSmall = cv2.resize(im, None, fx=1.0/FACE_DOWNSAMPLE_RATIO, fy=1.0/FACE_DOWNSAMPLE_RATIO, interpolation=cv2.INTER_LINEAR)

    rects = detector(imSmall, 0)
    if len(rects) == 0:
        return 1
    newRect = dlib.rectangle(int(rects[0].left()*FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].top()*FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].right()*FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].bottom()*FACE_DOWNSAMPLE_RATIO))

    points=[]
    [points.append((p.x, p.y)) for p in predictor(im, newRect).parts()]

    return points

capture = cv2.VideoCapture(0)

for i in range(50):
    ret, frame = capture.read()

totalTime = 0.0
validFrames = 0
dummyFrames = 50
spf = 0

while(validFrames < dummyFrames):
    validFrames +=1
    t = time.time()
    ret, frame = capture.read()
    height, width = frame.shape[:2]
    IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
    frame = cv2.resize(frame, None, fx=1.0/IMAGE_RESIZE, fy=1.0/IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)

    landmarks = getLandmarks(frame)
    timeLandmarks = time.time() -t

    # 未检测到人
    if landmarks == 1:
        validFrames -=1
        cv2.putText(frame, "Unable to detect face, Please check proper lighting", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Or Decrease FACE_DOWNSAMPLE_RATIO", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Blink Detection Demo", frame)

        if cv2.waitKey(1)&0xFF == 27:
            sys.exit()
    else:
        totalTime += timeLandmarks
        
spf = totalTime/dummyFrames
print("SPF为:{:.2f} ms".format(spf*1000))

drowsyLimit = drowsyTime/spf
falseBlinkLimit = blinkTime/spf
print("drowsyLimit:{}, falseBlinkLimit:{}".format(drowsyLimit, falseBlinkLimit))

while(1):
    try:
        t=time.time()
        ret, frame = capture.read()
        height, width = frame.shape[:2]
        IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
        frame = cv2.resize(frame, None, fx=1.0/IMAGE_RESIZE, fy=1.0/IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)
        landmarks= getLandmarks(frame)
        if landmarks==1:
            cv2.putText(frame, "Unable to detect face, Please check proper lighting", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Or Decrease FACE_DOWNSAMPLE_RATIO", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Blink Detection Demo", frame)

            if cv2.waitKey(1)&0xFF == 27:
                sys.exit()
            continue
        
        eyeStatus = checkEyeStatus(landmarks)
        checkBlinkStatus(eyeStatus)

        for i in range(0, len(leftEyeIndex)):
            cv2.circle(frame, (landmarks[leftEyeIndex[i]][0], landmarks[leftEyeIndex[i]][1]), 1, (255,0,255), thickness=1, lineType=cv2.LINE_AA)
        for i in range(0, len(rightEyeIndex)):
            cv2.circle(frame, (landmarks[rightEyeIndex[i]][0], landmarks[rightEyeIndex[i]][1]), 1, (255,0,255), thickness=1, lineType=cv2.LINE_AA)
        
        if(drowsy):
            cv2.putText(frame, "!!! Drowsy !!!", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        else:
            cv2.putText(frame, "Blink:{}".format(blinkCount), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            pass
        
        cv2.imshow("Blink Detection Demo", frame)
        
        if cv2.waitKey(1)&0xFF == 27:
            break
        print("耗时:", time.time() - t)
    
    except Exception as e:
        print(e)

capture.release()
cv2.destroyAllWindows()

    
