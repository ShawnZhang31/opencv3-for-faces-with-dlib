#
# @brief 提升Facial Landmark Detector的识别速度
# 
# @file fastWebcamFLD.py
# @author 张晓民
# @date 2018-08-24
#
import math
import sys
import cv2
import dlib
import numpy as np
from renderFace import renderFace

PREDICTOR_PATH = "../../common/resources/shape_predictor_68_face_landmarks.dat"
RESIZE_HEIGHT = 480
NUM_FRAMES_FOR_FPS = 100
SKIP_FRAMES = 2

# 计算眼睛外眼角的距离
def interEyeDistance(predict):
    leftEyeLeftCorner = (predict[36].x, predict[36].y)
    rightEyeRightCorner = (predict[45].x, predict[45].y)
    distance = cv2.norm(np.array(rightEyeRightCorner)-np.array(leftEyeLeftCorner))
    distance = int(distance)
    return distance



winName="Stablizing Facial Landmark Detector"
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

# 初始化摄像头对象
cap = cv2.VideoCapture(0)

# 检查是否成功打开摄像头
if(cap.isOpened() is False):
    print("打开摄像头失败")
    sys.exit()

print("摄像头打开成功")

winSize = 101
maxLevel = 10
fps=30.0

# 获取第一针图像
ret, imPrev=cap.read()

# 将当前帧转化为灰度图像
imGrayPrev = cv2.cvtColor(imPrev, cv2.COLOR_BGR2GRAY)

# 加载脸部探测器和形状评估器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# 初始化参数
points = [] #当前的landmark坐标
pointsPrev = [] #之前的landmark的坐标
pointsDetectedCur = [] #当前检测到的landmark的坐标
pointsDetectedPrev = [] #上一阵检测到的landmark的坐标

eyeDistanceNotCalculated = True
eyeDistance = 0
isFirstFrame = True

fps = 10
showStabilized = False
count = 0

while(True):
    if(count == 0):
        t = cv2.getTickCount()
    
    # 获取一帧数据
    ret, im = cap.read()
    # 将获取的数据转换为灰度图像
    imGray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    height = im.shape[0]
    IMAGE_RESIZE = float(height)/RESIZE_HEIGHT
    # 缩小图片使得追踪速度加快
    imSmall = cv2.resize(im,None, fx=1.0/IMAGE_RESIZE,fy=1.0/IMAGE_RESIZE,interpolation=cv2.INTER_LINEAR)

    # 跳帧达到加速的效果
    if(count%SKIP_FRAMES == 0):
        faces = detector(imSmall,0)
    
    # 检测是否检测到了脸部
    if len(faces) == 0:
        print("未检测到脸部")
    
    # 扫描每一张脸，检测器关键点
    else:
        for i in range(0,len(faces)):
            print("第%d张脸：" % (i+1))

            newRect = dlib.rectangle(int(faces[i].left()*IMAGE_RESIZE),
                int(faces[i].top()*IMAGE_RESIZE),
                int(faces[i].right()*IMAGE_RESIZE),
                int(faces[i].bottom()*IMAGE_RESIZE))
            
            # 第一帧要单独处理
            if(isFirstFrame):
                pointsPrev=[]
                pointsDetectedPrev=[]
                [pointsPrev.append((p.x, p.y)) for p in predictor(im, newRect).parts()]
                [pointsDetectedPrev.append((p.x, p.y)) for p in predictor(im, newRect).parts()]
            else:
                pointsPrev = []
                pointsDetectedPrev = []
                pointsPrev = points
                pointsDetectedPrev = pointsDetectedCur
            
            # pointsDetectedCur来储存用facial landmark detector检测的结果
            points = []
            pointsDetectedCur = []
            [points.append((p.x, p.y)) for p in predictor(im, newRect).parts()]
            [pointsDetectedCur.append((p.x, p.y)) for p in predictor(im, newRect).parts()]

            # 将numpy转化为float array
            pointsArr = np.array(points, np.float32)
            pointsPreArr = np.array(pointsPrev, np.float32)

            # 如果眼睛的距离
            if eyeDistanceNotCalculated:
                eyeDistance = interEyeDistance(predictor(im, newRect).parts())
                print("眼睛的距离:%d" % eyeDistance)
                eyeDistanceNotCalculated = False
            
            if eyeDistance > 100:
                dotRadius = 3
            else:
                dotRadius = 2
            
            sigma = eyeDistance*eyeDistance/400
            s = 2*int(eyeDistance/4)+1

            # 设置光流法参数
            lk_params = dict(winSize=(s,s), maxLevel=5, criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))

            pointsArr, status, err = cv2.calcOpticalFlowPyrLK(imGrayPrev, imGray, pointsPreArr, pointsArr, **lk_params)

            sigma = 100

            # 转换为float
            pointsArrFloat = np.array(points, np.float32)

            # 转化为list
            points = pointsArrFloat.tolist()

            for k in range(0, len(predictor(im, newRect).parts())):
                d=cv2.norm(np.array(pointsDetectedPrev[k]) - np.array(pointsDetectedCur[k]))
                alpha = math.exp(-d*d/sigma)
                points[k]=(1-alpha)*np.array(pointsDetectedCur[k])+alpha*np.array(points[k])
            
            # 绘制关键特征点
            if showStabilized is True:
                for p in points:
                    cv2.circle(im, (int(p[0]), int(p[1])), dotRadius, (255,0,0), -1)
            else:
                for p in pointsDetectedCur:
                    cv2.circle(im, (int(p[0]), int(p[1])), dotRadius, (0,0,255), -1)
            
            isFirstFrame = False
            count = count +1

            # 计算FPS
            if (count == NUM_FRAMES_FOR_FPS):
                t=(cv2.getTickCount()-t)/cv2.getTickFrequency()
                fps = NUM_FRAMES_FOR_FPS/t
                count = 0
                isFirstFrame=True

            cv2.putText(im, "{0:.2f}-fps".format(fps), (50, im.shape[0]-50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,255), 3, cv2.LINE_AA)
            cv2.imshow(winName, im)
            key = cv2.waitKey(25) & 0xFF

            if key == 32:
                showStabilized = not showStabilized
            
            if key == 27:
                sys.exit()
            

cv2.destroyAllWindows()
cap.release()



