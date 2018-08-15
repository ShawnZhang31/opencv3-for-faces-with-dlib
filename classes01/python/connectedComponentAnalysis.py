import cv2
import numpy as np

def displayConnectedComponents(im):
    imLables = im.copy()
    (minVal,maxVal,minLoc,maxLoc)=cv2.minMaxLoc(imLables)

    imLables=255*(imLables-minVal)/(maxVal-minVal)

    imLables=np.uint8(imLables)

    imColorMap=cv2.applyColorMap(imLables,cv2.COLORMAP_JET)

    cv2.imshow("Labels", imColorMap)
    cv2.waitKey(0)


im=cv2.imread("../../data/images/truth.png",cv2.IMREAD_GRAYSCALE)

th,imThresh=cv2.threshold(im,127,255,cv2.THRESH_BINARY)

_, imLabels=cv2.connectedComponents(imThresh)

displayConnectedComponents(imLabels)