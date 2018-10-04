import sys
import cv2
import dlib
import numpy as np
from renderFace import renderFace

SKIP_FRAMES = 5
RESIZE_HEIGHT = 320
PREDICTOR_PATH = "../../common/resources/zxm_shape_predictor_70_face_landmarks.dat"

def get3dModelPoints():
    """
    获取3D模型上的点
    Returns:
        np.array:3D模型点的数组
    """
    pass
    modelPoints = [[0.0, 0.0, 0.0],
                   [0.0, -330.0, -65.0],
                   [-225.0, 170.0, -135.0],
                   [225.0, 170.0, -135.0],
                   [-150.0, -150.0, -125.0],
                   [150.0, -150.0, -125.0]]
    return np.array(modelPoints, dtype=np.float64)

def get2dImagePoints(shape):
    """
    获取2D图像上的标记点
    Args:
        full_object_detection:dlib的识别结果
    Returns:
        np.array:2D图像上的标志点
    """
    pass
    imagePoints = [[shape.part(30).x, shape.part(30).y],
                   [shape.part(8).x, shape.part(8).y],
                   [shape.part(36).x, shape.part(36).y],
                   [shape.part(45).x, shape.part(45).y],
                   [shape.part(48).x, shape.part(48).y],
                   [shape.part(54).x, shape.part(54).y]]
    
    return np.array(imagePoints, dtype=np.float64)

def getCameraMatrix(flocalLength, center):
    """
    获取摄像头的变换矩阵
    Args:
        flocalLength:(float),摄像头的焦距;
        center:[x,y],焦点
    Returns:
        np.array
    """
    pass

    cameraMatrix = [[flocalLength, 0, center[0]],
                    [0, flocalLength, center[1]],
                    [0, 0, 1]]
    return np.array(cameraMatrix, dtype=np.float64)


if __name__=="__main__":
    try:
        capture = cv2.VideoCapture(0)
        if (capture.isOpened() is False):
            print("不能打开摄像头!")
            sys.exit(0)
        
        fps = 30.0
        ret, im = capture.read()
        if ret == True:
            height = im.shape[0]
            RESIZE_SCALE = float(height)/RESIZE_HEIGHT
            size = im.shape[0:2]
        else:
            print("不能读取视频中的内容")
            sys.exit(0)
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(PREDICTOR_PATH)

        t = cv2.getTickCount()
        count = 0

        while(True):
            if count ==0:
                t=cv2.getTickCount()
            
            ret, im=capture.read()
            imSmall = cv2.resize(im, None, fx=1.0/RESIZE_SCALE, fy=1.0/RESIZE_SCALE, interpolation=cv2.INTER_LINEAR)

            if(count % SKIP_FRAMES ==0):
                faces = detector(imSmall, 0)
                modelPoints = get3dModelPoints()

                for face in faces:
                    newRect = dlib.rectangle(int(face.left()*RESIZE_SCALE),
                                             int(face.top()*RESIZE_SCALE),
                                             int(face.right()*RESIZE_SCALE),
                                             int(face.bottom()*RESIZE_SCALE))
                    shape = predictor(im, newRect)
                    # renderFace(im, shape)

                    iamgePoints = get2dImagePoints(shape)

                    rows, cols, ch = im.shape
                    flocalLength = cols
                    cameraMatrix = getCameraMatrix(flocalLength, (rows/2, cols/2))

                    distCoeffs = np.zeros((4,1), dtype=np.float64)

                    success, rotationVector, translationVector = cv2.solvePnP(modelPoints, iamgePoints, cameraMatrix, distCoeffs)

                    noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
                    noseEndPoint2D, jacobain = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector, cameraMatrix, distCoeffs)

                    print("noseEndPoint2D:\n{}".format(noseEndPoint2D))
                    print("jacobain:\n{}".format(jacobain))

                    p1 = (int(iamgePoints[0,0]), int(iamgePoints[0,1]))
                    p2 = (int(noseEndPoint2D[0,0,0]), int(noseEndPoint2D[0,0,1]))

                    cv2.line(im, p1, p2, (100, 220, 2), thickness=2)

                
                
                cv2.imshow("webcam head pose", im)

                
            
            if(cv2.waitKey(1)&0xFF)==27:
                sys.exit()
            
            count = count+1

    except Exception as e:
        print(e)
    
    
    
