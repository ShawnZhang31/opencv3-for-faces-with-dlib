import cv2

# 读取文图像
source=cv2.imread("../../data/images/sample.jpg")

# 缩放因子
scaleX=0.6
scaleY=0.6

# 将图像缩小到原来的0.6
scaleDown=cv2.resize(source,None,fx=scaleX,fy=scaleY,interpolation=cv2.INTER_LINEAR)
# 将图像放大到原来的1.8倍
scaleUp=cv2.resize(source,None,fx=scaleX*3,fy=scaleY*3,interpolation=cv2.INTER_LINEAR)

# 裁切图像
crop=source[50:150,20:200]

# 显示所有的图像
cv2.imshow("Original",source)
cv2.imshow("Scaled Down",scaleDown)
cv2.imshow("Scaled Up",scaleUp)
cv2.imshow("Cropped Image",crop)

cv2.waitKey(0)