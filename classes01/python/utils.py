import cv2
import numpy as np

# 定义鼠标点击的监听事件
def mouseHandler(event, x ,y, flags, data):
    if event==cv2.EVENT_LBUTTONDOWN:
        cv2.circle(data['im'], (x,y), 2, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow("Image", data['im'])
        if len(data['points'])<4:
            data['points'].append([x,y])

# 定义获取四个特征点的方法
def get_four_points(im):
    data={}
    data['im']=im.copy()
    data['points']=[]

    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", mouseHandler, data)
    cv2.waitKey(0)

    points=np.vstack(data['points']).astype(float)

    return points