import cv2
import numpy as np

image = cv2.imread("../../data/images/mark.jpg")

# 绘制直线
imageLine = image.copy()
cv2.line(imageLine,(322,179),(400,183),(0,255,0),thickness=2,lineType=cv2.LINE_AA)
cv2.imshow("Line",imageLine)

# 绘制缘
imageCirle = image.copy()
cv2.circle(imageCirle,(350,200),150,color=(0,255,0),thickness=2,lineType=cv2.LINE_AA)
cv2.imshow("Circle",imageCirle)

# 绘制椭圆:注意-椭圆的中心点和轴长一定要用整数
imageEllipse=image.copy()
cv2.ellipse(imageEllipse,(208,55),(450,355),45,0,300,(255,0,0),thickness=2,lineType=cv2.LINE_AA)
cv2.imshow("Ellipse",imageEllipse)

# 绘制矩形
imageRectangle = image.copy()
cv2.rectangle(imageRectangle,(208,55),(450,355),(0,255,0),thickness=2,lineType=cv2.LINE_8)
cv2.imshow("Rectangle",imageRectangle)

# 书写文字
imageText = image.copy()
cv2.putText(imageText,"Mark Zuckerberg",(205,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
cv2.imshow("Text",imageText)

cv2.waitKey(0)
