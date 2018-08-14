import cv2
import numpy as np
# 读取一个文件
source=cv2.imread("../../data/images/sample.jpg")

scaleFactor=1/255.0

# 将unsigned int 转为float
source=np.float32(source)
source=source*scaleFactor

cv2.imshow("float",source)
# 将float转为unsigned char
source=source*(1.0/scaleFactor)
source=np.uint8(source)
cv2.imshow("unsigned char",source)

cv2.waitKey(0)