import cv2
import numpy as np

# 前景图
# 读取含有alpha通道的图片需要使用IMREAD_UNCHANGED标志位
foreGroundImage = cv2.imread("../data/images/foreGroundAssetLarge.png", cv2.IMREAD_UNCHANGED)

# 拆分通道
b,g,r,a=cv2.split(foreGroundImage)

# 合并BGR
foreground = cv2.merge((b,g,r))
alpha = cv2.merge((a,a,a))

# 背景图
background = cv2.imread("../data/images/backGroundLarge.jpg")

# 把图像转化为8浮点型
foreground = foreground.astype(float)
background = background.astype(float)
alpha = alpha.astype(float)/255.0

foreground = cv2.multiply(alpha, foreground)
background = cv2.multiply(1.0-alpha, background)
outImage = cv2.add(foreground, background)

cv2.imshow("outImage", outImage/255)
cv2.waitKey(0)
