import cv2
import numpy as np
import dlib

def findIndex(points, point):
    """
    获取点的索引
    参数
    ---------
    points:[]
        点集
    point:[] 
        要查询的点
    返回:
    ---------
    int:
        返回要查询的顶点的索引
    """
    pass

    diff = np.array(points) - np.array(point)
    # 获得当前的点距离所有的距离
    diffNorm = np.linalg.norm(diff, 2, 1)
    print("diffNorm:{}".format(diffNorm))

    # 找出最小值并返回索引
    return np.argmin(diffNorm)


def writeDelaunay(subdiv, points, outputFilename):
    """
    将识别出来的Delaunay三角形的三角形顶点列表写入文件
    参数
    ---------
    subdiv:cv2.Subdiv2D
        OpenCV三角划分类
    points:[]
        三角形顶点列表
    outputFilename:string
        输出文件的列表
    返回
    ---------
    无返回值
    """
    pass
    
    triangleList = subdiv.getTriangleList()
    filePointer = open(outputFilename, 'w')

    for t in triangleList:
        pt1=(t[0], t[1])
        pt2=(t[2], t[3])
        pt3=(t[4], t[5])

        landmark1 = findIndex(points, pt1)
        landmark2 = findIndex(points, pt2)
        landmark3 = findIndex(points, pt3)

        filePointer.writelines("{} {} {}\n".format(landmark1, landmark2, landmark3))
    
    # 写入完毕关闭文件
    filePointer.close()

def faceLandmarksDetect(im):
    """
    获取脸部的关键特征点
    参数
    ---------
    im:需要检测的图像
    返回值
    ---------
    points:检测到的关键点
    """
    pass

    PREDICTOR_PATH="../../common/resources/shape_predictor_68_face_landmarks.dat"
    faceDetector = dlib.get_frontal_face_detector()
    shapePredictor = dlib.shape_predictor(PREDICTOR_PATH)

    faceRects=faceDetector(im,0)
    points=[]
    if len(faceRects)>0:
        newRect=dlib.rectangle(int(faceRects[0].left()),int(faceRects[0].top()),int(faceRects[0].right()),int(faceRects[0].bottom()))
        landmarks=shapePredictor(im,newRect)
        for p in landmarks.parts():
            points.append((p.x, p.y))
    print("检测到的脸部关键点为:")
    for index, value in enumerate(points):
        print("{}:{}".format(index, value))
    
    return points



# 入口
if __name__=="__main__":
    # 显示图像的窗口的名称
    win = "Delaunay Tiangulation & Voronoi Diagram"

    # 定义绘制的颜色
    delaunayColor = (255,255,255)
    pointsColor = (0,0,255)
    # 识别图
    img = cv2.imread("../data/images/smiling-man.jpg")
    size = img.shape
    rect = (0,0,size[1],size[0])

    subdiv = cv2.Subdiv2D(rect)

    # 图片中的面部关键点
    points=faceLandmarksDetect(img)
    
    # 输出文件的名称
    outputFileName = "results/smiling-man-delaunay.tri"

    # 将关键点输入划分器
    for p in points:
        subdiv.insert(p)
    
    writeDelaunay(subdiv, points, outputFileName)
