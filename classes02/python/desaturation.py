import cv2
import numpy as np

img=cv2.imread("../data/images/capsicum.jpg")
saturationScale=0.1
hsvImage=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

hsvImage=np.float32(hsvImage)

H,S,V=cv2.split(hsvImage)
S=np.clip(S*saturationScale,0,255)

hsvImage=np.uint8(cv2.merge([H,S,V]))
imSat=cv2.cvtColor(hsvImage,cv2.COLOR_HSV2BGR)

combined=np.hstack([img,imSat])
cv2.imshow("Combined",combined)
cv2.waitKey(0)
cv2.destroyAllWindows()