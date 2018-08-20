#
# @brief 提升Facial Landmark Detector的识别速度
# 
# @file fastWebcamFLD.py
# @author 张晓民
# @date 2018-08-20
#
import sys
import cv2
import dlib
import numpy as np
from renderFace import renderFace

PREDICTOR_PATH="../../common/resources/shape_predictor_68_face_landmarks.dat"
RESIZE_HEIGHT=480
SKIP_FRAMES=2

winName="Fast Facial Landmark Detector"
cap = cv2.VideoCapture(0)

# 检查是否成功打开摄像头
if(cap.isOpened() is False):
    print("打开摄像头失败")
    sys.exit()

print("摄像头打开成功")

fps=30.0

# 获取第一针图像
ret, im=cap.read()

if ret == True:
    height=im.shape[0]
    RESIZE_SCALE=float(height)/RESIZE_HEIGHT
    size=im.shape[0:2]
else:
    print("读取图片失败")
    sys.exit()

# 加载脸部探测器和形状评估器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# 初始化计时器
t=cv2.getTickCount()
count = 0

# 开始处理
while(True):
    if count==0:
        t=cv2.getTickCount()

    ret, im=cap.read()

    # 将图片缩小用于追踪
    imSmall=cv2.resize(im,None,fx=1.0/RESIZE_SCALE,fy=1.0/RESIZE_SCALE,interpolation=cv2.INTER_LINEAR)
    if(count%SKIP_FRAMES == 0):
        faces=detector(imSmall,0)
    for face in faces:
        newRect= dlib.rectangle(int(face.left()*RESIZE_SCALE),
                                int(face.top()*RESIZE_SCALE),
                                int(face.right()*RESIZE_SCALE),
                                int(face.bottom()*RESIZE_SCALE))
        shap = predictor(im, newRect)
        renderFace(im, shap)
    
    cv2.putText(im, "{0:.2f}-fps".format(fps), (50, size[0]-50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,255),2)
    cv2.imshow(winName, im)

    key=cv2.waitKey(1) & 0xFF

    if key==27:
        sys.exit()
    
    count = count+1
    
    print("t1:",cv2.getTickCount())
    print("t2:",cv2.getTickFrequency())

    if(count==100):
        t=(cv2.getTickCount()-t)/cv2.getTickFrequency()
        fps=100.0/t
        count=0

# 退出释放内存
cap.release()
cv2.destroyAllWindows()

