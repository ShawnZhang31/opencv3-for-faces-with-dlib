import cv2

# 读入图像
source = cv2.imread("../../data/images/threshold.png",cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", source)

# 定义阈值和最大值
thresh=0
maxValue=255

# 阈值化处理
ret,source=cv2.threshold(source,thresh,maxValue,cv2.THRESH_BINARY)
cv2.imshow("Threshold",source)

cv2.waitKey()
cv2.destroyAllWindows()