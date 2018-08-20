import cv2
import numpy as np

def drawPolyline(im, landmarks, start, end, isClosed=False):
    """ 根据检测到关键特征点来绘制特征点连线
    Args:
        im: 绘制的图片
        landmark: 关键特征点
        start: 起始位置的点的索引
        end: 结束位置的点的索引
        isClosed: 连线是否封闭，默认是封闭的
    """
    points=[]

    for i in range(start, end+1):
        point=[landmarks.part(i).x, landmarks.part(i).y]
        points.append(point)
    
    points=np.array(points, dtype=np.int32)
    cv2.polylines(im,[points], isClosed, (255,200,0), thickness=2, lineType=cv2.LINE_AA)

def renderFace(im, landmarks):
    """ 根据脸部关键特征点绘制特征线
    Args:
        im: 绘制的图片
        landmarks: 关键特征点l列表
    """
    # assert(landmarks.num_parts ==68)    #判断是否是68个点
    drawPolyline(im, landmarks, 0, 16)           # 线板
    drawPolyline(im, landmarks, 17, 21)          # 左眉毛
    drawPolyline(im, landmarks, 22, 26)          # 右眉毛
    drawPolyline(im, landmarks, 27, 30)          # 鼻梁
    drawPolyline(im, landmarks, 30, 35, True)    # 鼻尖
    drawPolyline(im, landmarks, 36, 41, True)    # 左眼眶
    drawPolyline(im, landmarks, 42, 47, True)    # 右眼眶
    drawPolyline(im, landmarks, 48, 59, True)    # 外嘴唇
    drawPolyline(im, landmarks, 60, 67, True)    # 右嘴唇
