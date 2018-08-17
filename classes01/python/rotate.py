import cv2

source=cv2.imread("../../data/images/sample.jpg")

# 获取图片的尺寸
dim=source.shape
M=cv2.getRotationMatrix2D((dim[1]/2, dim[0]/2),-30,1.0)
dst=cv2.warpAffine(source,M,(dim[1],dim[0]))

cv2.imshow("Original", source)
cv2.imshow("Rotate", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()