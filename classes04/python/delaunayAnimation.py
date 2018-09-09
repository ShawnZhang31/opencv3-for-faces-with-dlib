"""
delaunayAnimation.py
==========
@brief 使用动画展示Delaunay三角剖分和Voronoi的划分的过程

@file delaunayAnimation.cpp
@author Shawn Zhang
@date 2018-09-09
"""
import cv2
import dlib
import numpy as np
import random

def rectContains(rect, point):
    """
    检查点是否在矩形内部
    参数
    --------------
    rect:矩形 [x,y,width,height]
    point:点 [x,y]
    返回值
    --------------
    True -> 点在矩形内部
    False -> 点在矩形外
    """
    pass

    if point[0]<rect[0]:
        return False
    elif point[1]<rect[1]:
        return False
    elif point[0]>(rect[2]+rect[0]):
        return False
    elif point[1]>(rect[3]+rect[1]):
        return False
    return True

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

    PREDICTOR_PATH="../../common/resources/zxm_shape_predictor_70_face_landmarks.dat"
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

def drawPoint(img, p, color):
    """
    绘制点
     参数
     -----------
    img:绘制的图像;
    p: 点坐标;
    color:绘制用的颜色
    """
    pass

    cv2.circle(img, p, 2, color)

def drawDelaunay(img, subdiv, delaunayColor):
    """
    绘制Delaunay剖分三角形
    参数
    ---------
    img:绘制的图像;
    subdiv:剖分器;
    delaunayColor:绘制用的颜色
    """
    pass

    triangleList = subdiv.getTriangleList()
    size = img.shape
    r=(0,0,size[1],size[0])

    for t in triangleList:
        pt1 = (t[0],t[1])
        pt2 = (t[2],t[3])
        pt3 = (t[4],t[5])

        if rectContains(r, pt1) and rectContains(r, pt2) and rectContains(r, pt3):
            cv2.line(img, pt1, pt2, delaunayColor)
            cv2.line(img, pt2, pt3, delaunayColor)
            cv2.line(img, pt3, pt1, delaunayColor)

def drawVoronoi(img, subdiv):
    """
    绘制Voronoi图
    参数
    ------------
    img:绘制的画板;
    subdiv:剖分器
    """
    pass

    (facets, centers)=subdiv.getVoronoiFacetList([])

    for i in range(0, len(facets)):
        ifaceArr = []
        for f in facets[i]:
            ifaceArr.append(f)
        
        ifacet = np.array(ifaceArr, np.int)

        # 生成随机颜色
        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))

        # 绘制面
        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0)

        # 绘制边界
        ifacets=np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0,0,0), 1, cv2.LINE_AA, 0)

        # 绘制中心
        cv2.circle(img, (centers[i][0], centers[i][1]), 2, (0,0,0), -1, cv2.LINE_AA, 0)

# 入口
if __name__ == "__main__":
    win="Delaunay Animation"
    delaunayColor = (255, 255, 255)
    pointsColor = (0, 0, 255)

    img = cv2.imread("../data/images/smiling-man.jpg")

    rect=(0, 0, img.shape[1], img.shape[0])
    subdiv = cv2.Subdiv2D(rect)

    # 获取脸部的关键点
    points = faceLandmarksDetect(img)

    imgVoronoi = np.zeros(img.shape, dtype=img.dtype)
    imgDelaunay = img.copy()

    # 绘制
    plotPoints = []
    for p in points:
        drawPoint(img, p, pointsColor)
        subdiv.insert(p)
        plotPoints.append(p)

        imgDelaunay = img.copy()

        drawDelaunay(imgDelaunay, subdiv, delaunayColor)
        drawVoronoi(imgVoronoi,subdiv)

        imDisplay = np.hstack([imgDelaunay, imgVoronoi])
        cv2.imshow(win, imDisplay)
        cv2.waitKey(400)
    

    cv2.waitKey(0)
    cv2.destoryAllWindows()
    
