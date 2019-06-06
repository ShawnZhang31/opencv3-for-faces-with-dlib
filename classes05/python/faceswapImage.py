import sys, cv2, dlib, time
import numpy as np
import faceBlendCommon as fbc


if __name__ == "__main__":
    model_path = "../../common/resources/shape_predictor_68_face_landmarks.dat"

    # 初始化dlib检测器
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    # 计算用时
    t = time.time()

    # 要处理的图片
    filename1 = "../data/images/ted_cruz.jpg"
    filename2 = "../data/images/donald_trump.jpg"
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    img1Warped = np.copy(img2)

    # 关键点检测
    points1 = fbc.getLandmarks(detector, predictor, img1)
    points2 = fbc.getLandmarks(detector, predictor, img2)

    # 检测外框
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints= False)
    for i in range(0, len(hullIndex)):
        hull1.append(points1[hullIndex[i][0]])
        hull2.append(points2[hullIndex[i][0]])
    
    sizeImg2 = img2.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])
    dt = fbc.calculateDelaunayTriangles(rect, hull2)
    
    if len(dt) == 0:
        quit()
    
    # 仿射变化
    for i in range(0, len(dt)):
        t1 = []
        t2 = []
        
        for j in range(0,3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])

        fbc.warpTriangle(img1, img1Warped, t1, t2)
    
    print("faceswap花费{:.3f}秒".format(time.time()-t))

    tClone = time.time()
    # 在进行无缝拷贝的时候需要先创建一个模板
    mask = np.zeros(img2.shape, dtype=img2.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull2), (255, 255, 255))
    r = cv2.boundingRect(np.array([hull2]))
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
    output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
    print("无缝拷贝耗时{:.3f}秒".format(time.time()-tClone))
    print("总共耗时{:.3f}秒".format(time.time()-t))
    cv2.imshow("no seamless", np.uint8(img1Warped))
    cv2.imshow("seamless", np.uint8(output))
    cv2.waitKey(0)