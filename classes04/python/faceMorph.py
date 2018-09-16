import faceBlendCommon as fbc
import sys
import cv2
import dlib
import numpy as np

# 给lamda调取用
def morphImages(alpha, imgNorm1, postions1, imgNorm2, postions2, dt):
    pointsMorph = (1.0-alpha)*points1 + alpha*points2

    imOut1 = fbc.warpImage(imgNorm1, points1, pointsMorph, dt)
    imOut2 = fbc.warpImage(imgNorm2, points2, pointsMorph, dt)

    imMorph = (1-alpha)*imOut1 + alpha*imOut2

    imgShow=np.zeros((480,480*3,3),dtype=float)
    imgShow=cv2.hconcat([imgNorm1, imgNorm2, imMorph])

    cv2.imshow(winName, imgShow)


if __name__ == "__main__":
    # 加载识别器
    PREDICTOR_DIR ="../../common/resources/zxm_shape_predictor_70_face_landmarks.dat"
    faceDetector = dlib.get_frontal_face_detector()
    landmarkDetector = dlib.shape_predictor(PREDICTOR_DIR)

    # 加载基本图片
    img1 = cv2.imread("../data/images/girls/xiaohuizi.jpg")
    img2 = cv2.imread("../data/images/girls/hexiaoping.jpg")

    # 获取关键特征点
    points1 = fbc.getLandmarks(faceDetector, landmarkDetector, img1)
    points2 = fbc.getLandmarks(faceDetector, landmarkDetector, img2)

    points1=np.array(points1)
    points2=np.array(points2)

    img1=np.float32(img1)/255.0
    img2=np.float32(img2)/255.0

    h=480
    w=480

    imgNorm1, points1 = fbc.normalizeImagesAndLandmarks((h,w), img1, points1)
    imgNorm2, points2 = fbc.normalizeImagesAndLandmarks((h,w), img2, points2)

    pointsAvg = (points1+points2)/2.0

    # 边界点
    boundaryPoints = fbc.getEightBoundaryPoints(h, w)
    points1 = np.concatenate((points1, boundaryPoints), axis=0)
    points2 = np.concatenate((points2, boundaryPoints), axis=0)
    pointsAvg = np.concatenate((pointsAvg, boundaryPoints), axis=0)

    # 就算细分三角形
    rect = (0,0,w,h)
    dt = fbc.calculateDelaunayTriangles(rect, pointsAvg)

    winName = "Face Morphing"
    
    cv2.namedWindow(winName,cv2.WINDOW_AUTOSIZE)
    #  回调函数使用lambda语法
    cv2.createTrackbar("alpha",winName,0,10,lambda v:morphImages(v/10.0, imgNorm1, points1, imgNorm2, points2, dt))
    
    alpha = cv2.getTrackbarPos("alpha", winName)

    morphImages(alpha/10.0, imgNorm1, points1, imgNorm2, points2, dt)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


